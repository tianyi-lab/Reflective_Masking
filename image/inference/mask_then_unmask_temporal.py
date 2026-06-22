
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SYSTEM_PROMPT = (
    "Generate an image applying the following editing instruction based on the original image."
)


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------


def build_model_instruction(system_prompt: str, user_text: str) -> str:
    return f"<system>{system_prompt}</system><user>{user_text}</user>"


def _as_list(values):
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def build_image_tokens_from_before(before_ids, vq_offset: int, newline_token: int):
    rows = _as_list(before_ids)
    if not rows or not isinstance(rows[0], list):
        raise ValueError("before_ids must be a 2D array-like object")

    h = len(rows)
    w = len(rows[0])
    img_tokens = []
    for row in rows:
        if len(row) != w:
            raise ValueError("before_ids rows must have consistent width")
        img_tokens.extend(int(token) + int(vq_offset) for token in row)
        img_tokens.append(int(newline_token))
    return img_tokens, h, w, h * w


def build_content_positions(image_start: int, h: int, w: int) -> list[int]:
    out = []
    for r in range(h):
        row_base = image_start + r * (w + 1)
        for c in range(w):
            out.append(row_base + c)
    return out


def _token_ids_to_list(tokenized) -> list[int]:
    ids = tokenized.input_ids[0]
    if hasattr(ids, "tolist"):
        return ids.tolist()
    return list(ids)


def build_full_sequence(
    tokenizer,
    system_prompt: str,
    instruction: str,
    img_tokens: list[int],
    *,
    answer_begin: int,
    boi_token: int,
    eoi_token: int,
    answer_end: int,
) -> tuple[list[int], int, int, int]:
    inst_text = build_model_instruction(system_prompt=system_prompt, user_text=instruction)
    inst_ids = _token_ids_to_list(
        tokenizer(
            inst_text,
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors="pt",
        )
    )
    full_seq = inst_ids + [answer_begin] + [boi_token] + img_tokens + [eoi_token] + [answer_end]
    image_start = len(inst_ids) + 2
    image_end = len(full_seq) - 2
    cfg_tail_start = len(inst_ids)
    return full_seq, image_start, image_end, cfg_tail_start


def build_uncond_prefix(tokenizer, system_prompt: str, uncond_prompt: str) -> list[int]:
    uncond_text = build_model_instruction(system_prompt=system_prompt, user_text=uncond_prompt)
    return _token_ids_to_list(
        tokenizer(
            uncond_text,
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors="pt",
        )
    )


def resolve_tokenizer_path(checkpoint: str, tokenizer_path: str) -> str:
    if tokenizer_path:
        return tokenizer_path
    ckpt_path = Path(checkpoint)
    if (ckpt_path / "tokenizer.json").is_file() or (ckpt_path / "tokenizer_config.json").is_file():
        return str(ckpt_path)

    config_path = ckpt_path / "config.json"
    if config_path.is_file():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config_data = {}
        source = str(config_data.get("_name_or_path") or "").strip()
        if source:
            source_path = Path(source)
            if (source_path / "tokenizer.json").is_file() or (source_path / "tokenizer_config.json").is_file():
                return source
    return checkpoint


# ---------------------------------------------------------------------------
# Prompt / instruction resolution
# ---------------------------------------------------------------------------


def clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def manifest_keys_from_row(row: dict) -> list[str]:
    keys: list[str] = []
    source = clean_text(row.get("source"))
    sample_id = clean_text(row.get("sample_id"))
    if source and sample_id:
        keys.append(f"{source}/{sample_id}")

    for field_name in ("sample_dir", "instruction_path"):
        raw_value = clean_text(row.get(field_name))
        if not raw_value:
            continue
        path_value = raw_value.replace("\\", "/").strip("/")
        if field_name == "instruction_path":
            path_value = str(Path(path_value).parent).replace("\\", "/").strip("/")
        if path_value:
            keys.append(path_value)
            parts = path_value.split("/")
            if len(parts) >= 2:
                keys.append("/".join(parts[-2:]))
    return unique(keys)


def sample_keys(sample_dir: Path, data_root: Path | None) -> list[str]:
    keys = [f"{sample_dir.parent.name}/{sample_dir.name}"]
    if data_root is not None:
        try:
            keys.insert(0, sample_dir.relative_to(data_root).as_posix())
        except ValueError:
            pass
    return unique(keys)


def load_instruction_manifest(manifest_path: Path, instruction_field: str):
    from collections import Counter

    if not manifest_path.is_file():
        raise FileNotFoundError(f"instruction manifest not found: {manifest_path}")

    instructions: dict[str, str] = {}
    empty_keys: set[str] = set()
    stats: Counter = Counter()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            stats["rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json in {manifest_path}:{line_no}: {exc}") from exc

            keys = manifest_keys_from_row(row)
            if not keys:
                stats["missing_key"] += 1
                continue

            instruction = clean_text(row.get(instruction_field))
            if not instruction:
                stats["empty_instruction"] += 1
                empty_keys.update(keys)
                continue

            stats["usable_rows"] += 1
            for key in keys:
                if key in instructions:
                    stats["duplicate_keys"] += 1
                    continue
                instructions[key] = instruction
                empty_keys.discard(key)
    return instructions, empty_keys, dict(sorted(stats.items()))


def find_manifest_instruction(
    *,
    sample_dir: Path,
    data_root: Path | None,
    instructions: dict[str, str],
    empty_keys: set[str],
) -> tuple[str | None, str, list[str]]:
    keys = sample_keys(sample_dir, data_root)
    for key in keys:
        instruction = instructions.get(key)
        if instruction:
            return instruction, "ok", keys
    for key in keys:
        if key in empty_keys:
            return None, "empty_generation_instruction", keys
    return None, "missing_manifest_row", keys


def read_file_prompt(sample_dir: Path, instruction_file: str) -> str:
    instruction_path = sample_dir / instruction_file
    if not instruction_path.is_file():
        raise FileNotFoundError(f"missing instruction file: {instruction_path}")
    text = instruction_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"^Prompt:\s*", "", text).strip()
    if not text:
        raise ValueError(f"empty instruction file: {instruction_path}")
    return text


def resolve_prompt(
    *,
    args,
    sample_dir: Path,
    data_root: Path | None,
    manifest_instructions: dict[str, str],
    empty_manifest_keys: set[str],
) -> tuple[str, dict]:
    if args.prompt:
        prompt = args.prompt.strip()
        if not prompt:
            raise ValueError("--prompt is empty")
        return prompt, {"source": "override", "field": "", "manifest": "", "reason": "ok", "keys": []}

    if args.instruction_source == "manifest":
        prompt, reason, keys = find_manifest_instruction(
            sample_dir=sample_dir,
            data_root=data_root,
            instructions=manifest_instructions,
            empty_keys=empty_manifest_keys,
        )
        if prompt is None:
            raise ValueError(
                f"manifest instruction unavailable for sample={sample_dir} "
                f"reason={reason} keys={keys}"
            )
        return prompt, {
            "source": "manifest",
            "field": str(args.instruction_field),
            "manifest": str(args.instruction_manifest),
            "reason": reason,
            "keys": keys,
        }

    prompt = read_file_prompt(sample_dir, args.instruction_file)
    return prompt, {
        "source": "file",
        "field": str(args.instruction_file),
        "manifest": "",
        "reason": "ok",
        "keys": sample_keys(sample_dir, data_root),
    }


def _read_sample_list(path: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"invalid sample-list line {line_no}: expected '<sample_dir>\\t<output_dir>'")
            pairs.append((Path(parts[0]), Path(parts[1])))
    return pairs


# ---------------------------------------------------------------------------
# Mask / image I/O
# ---------------------------------------------------------------------------


def mask_grid_to_indices(mask_grid) -> object:
    import numpy as np

    rows, cols = np.nonzero(mask_grid.astype(bool))
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([rows, cols], axis=1).astype(np.int64, copy=False)


def load_mask_grid(sample_dir: Path, field: str, h: int, w: int):
    import numpy as np

    path = sample_dir / f"{field}.npy"
    if not path.is_file():
        return None, path
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 2 and arr.shape == (h, w):
        return arr.astype(bool), path
    if arr.ndim == 2 and arr.shape[1] == 2:
        coords = arr.astype(np.int64, copy=False)
        grid = np.zeros((h, w), dtype=bool)
        if coords.size:
            rows = coords[:, 0]
            cols = coords[:, 1]
            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            grid[rows[valid], cols[valid]] = True
        return grid, path
    if arr.ndim == 1 and arr.size == h * w:
        return arr.reshape(h, w).astype(bool), path
    raise ValueError(
        f"unsupported GT mask shape {arr.shape} in {path}; expected Nx2, "
        f"{(h, w)} mask, or flat length {h * w}"
    )


def save_vq_ids_image(
    *,
    ids_hw,
    output_path: Path,
    model_device,
    vq_offset: int,
    vae_scale: int,
    vqvae,
    vae_ckpt: str,
    decode_vq_to_image,
) -> None:
    import numpy as np
    import torch

    h, w = map(int, ids_hw.shape)
    tokens = torch.tensor(
        (ids_hw.reshape(-1).astype(np.int64) + int(vq_offset)).tolist(),
        dtype=torch.long,
        device=model_device,
    ).view(1, h * w)
    decode_vq_to_image(
        tokens,
        str(output_path),
        vae_ckpt=vae_ckpt,
        image_height=h * vae_scale,
        image_width=w * vae_scale,
        vqvae=vqvae,
    ).save(output_path)


def save_mask_preview(
    *,
    before_ids,
    pred_grid,
    output_path: Path,
    model_device,
    vq_offset: int,
    vae_scale: int,
    vqvae,
    vae_ckpt: str,
    decode_vq_to_image,
) -> None:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    h, w = map(int, before_ids.shape)
    before_tokens = torch.tensor(
        (before_ids.reshape(-1).astype(np.int64) + int(vq_offset)).tolist(),
        dtype=torch.long,
        device=model_device,
    ).view(1, h * w)
    image = decode_vq_to_image(
        before_tokens,
        str(output_path),
        vae_ckpt=vae_ckpt,
        image_height=h * vae_scale,
        image_width=w * vae_scale,
        vqvae=vqvae,
    ).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    mask = pred_grid.astype(bool)
    for rr, cc in zip(*mask.nonzero()):
        x0 = int(cc) * vae_scale
        y0 = int(rr) * vae_scale
        x1 = min(image.size[0], x0 + vae_scale) - 1
        y1 = min(image.size[1], y0 + vae_scale) - 1
        draw.rectangle((x0, y0, x1, y1), fill=(255, 0, 0, 110))
    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)


# ---------------------------------------------------------------------------
# Two-stage mask -> unmask inference
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage temporal mask->unmask inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Temporal checkpoint path")
    parser.add_argument("--sample-dir", type=str, default="", help="Offline cache sample directory")
    parser.add_argument(
        "--sample-list",
        type=str,
        default="",
        help="Optional TSV file with sample_dir, output_dir pairs. Loads the model once.",
    )
    parser.add_argument("--tokenizer-path", type=str, default="", help="Tokenizer path")
    parser.add_argument("--data-root", type=str, default="", help="Cache root used for manifest key matching")
    parser.add_argument("--instruction-file", type=str, default="instruction.txt")
    parser.add_argument(
        "--instruction-source",
        type=str,
        default="file",
        choices=("file", "manifest"),
        help="file reads instruction.txt; manifest reads --instruction-field from --instruction-manifest",
    )
    parser.add_argument("--instruction-manifest", type=str, default="", help="JSONL prompt manifest")
    parser.add_argument("--instruction-field", type=str, default="generation_instruction")
    parser.add_argument("--prompt", type=str, default="", help="Override prompt/instruction text")
    parser.add_argument("--vae-ckpt", type=str, default="./vae_ckpt", help="Lumina VAE checkpoint")
    parser.add_argument("--output-dir", type=str, default="results_temporal_mask_then_unmask")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--uncond-prompt", type=str, default="<uncondition>")
    parser.add_argument("--mask-timesteps", type=int, default=8)
    parser.add_argument("--unmask-timesteps", type=int, default=8)
    parser.add_argument(
        "--mask-confidence-threshold",
        type=float,
        default=0.9,
        help="Phase-1: select token iff p(MASK) > threshold and p(MASK) > p(current token)",
    )
    parser.add_argument(
        "--unmask-confidence-threshold",
        type=float,
        default=0.9,
        help="Phase-2 threshold mode only: unmask iff p(best VQ) > threshold and p(best VQ) > p(MASK)",
    )
    parser.add_argument(
        "--unmask-decode-mode",
        type=str,
        default="lumina",
        choices=("lumina", "threshold"),
        help="Phase-2 decode rule. lumina uses the original Lumina/MaskGIT cosine schedule.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Lumina/MaskGIT sampling temperature.")
    parser.add_argument(
        "--final-unmask-flush",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force any remaining MASK tokens to VQ argmax after phase 2.",
    )
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--temporal-history-decay", type=float, default=0.2)
    parser.add_argument("--temporal-history-scale", type=float, default=1.0)
    parser.add_argument(
        "--mask-temporal-history-scale",
        type=float,
        default=None,
        help="Phase-1 history scale. Defaults to --temporal-history-scale.",
    )
    parser.add_argument(
        "--unmask-temporal-history-scale",
        type=float,
        default=None,
        help="Phase-2 history scale. Defaults to --temporal-history-scale.",
    )
    parser.add_argument(
        "--temporal-history-mode",
        type=str,
        default="gated_post_rms",
        choices=("post_rms", "gated_post_rms", "residual_norm", "gated_residual"),
    )
    parser.add_argument("--temporal-history-gate-min", type=float, default=0.002)
    parser.add_argument("--temporal-history-gate-max", type=float, default=0.04)
    parser.add_argument("--temporal-history-max-steps", type=int, default=6)
    parser.add_argument("--gt-mask-field", type=str, default="edit_indices_original_dataset")
    parser.add_argument(
        "--save-mask-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save phase-1 predicted mask preview.",
    )
    parser.add_argument(
        "--save-debug-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save two_stage_debug.json.",
    )
    parser.add_argument(
        "--save-step-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-step decoded images under debug_steps/.",
    )
    parser.add_argument(
        "--save-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save GT mask and after_ids decode when available.",
    )
    return parser.parse_args()


def _safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_") or "step"


def save_state_preview(
    *,
    state,
    before_flat_tokens,
    output_path: Path,
    h: int,
    w: int,
    mask_token: int,
    newline_token: int,
    model_device,
    vae_scale: int,
    vqvae,
    vae_ckpt: str,
    decode_vq_to_image,
) -> int:
    import torch
    from PIL import ImageDraw

    seq_len = int(h) * int(w)
    flat = state[state != int(newline_token)].view(1, seq_len)
    mask_flat = flat == int(mask_token)
    display_flat = torch.where(mask_flat, before_flat_tokens.cpu(), flat).to(model_device)
    frame = decode_vq_to_image(
        display_flat,
        str(output_path),
        vae_ckpt=vae_ckpt,
        image_height=int(h) * int(vae_scale),
        image_width=int(w) * int(vae_scale),
        vqvae=vqvae,
    )
    mask_count = int(mask_flat.sum().item())
    if mask_count:
        draw = ImageDraw.Draw(frame)
        mask_grid = mask_flat.view(h, w)
        for rr, cc in mask_grid.nonzero(as_tuple=False).tolist():
            x0 = int(cc) * int(vae_scale)
            y0 = int(rr) * int(vae_scale)
            x1 = x0 + int(vae_scale) - 1
            y1 = y0 + int(vae_scale) - 1
            draw.rectangle((x0, y0, x1, y1), fill=(255, 0, 0))
    frame.save(output_path)
    return mask_count


def save_step_debug_images(
    *,
    debug_info: dict,
    output_dir: Path,
    before_flat_tokens,
    h: int,
    w: int,
    mask_token: int,
    newline_token: int,
    model_device,
    vae_scale: int,
    vqvae,
    vae_ckpt: str,
    decode_vq_to_image,
) -> list[dict]:
    steps_dir = output_dir / "debug_steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    states = debug_info.get("states", [])
    records = debug_info.get("records", [])
    saved = []
    for idx, state in enumerate(states):
        record = records[idx] if idx < len(records) else {}
        phase = _safe_stem(str(record.get("phase", "phase")))
        name = _safe_stem(str(record.get("name", "step")))
        step = int(record.get("step", idx))
        step_label = f"{step:04d}" if step >= 0 else f"neg{abs(step):04d}"
        path = steps_dir / f"{idx:04d}_{phase}_{name}_{step_label}.png"
        mask_count = save_state_preview(
            state=state,
            before_flat_tokens=before_flat_tokens,
            output_path=path,
            h=h,
            w=w,
            mask_token=mask_token,
            newline_token=newline_token,
            model_device=model_device,
            vae_scale=vae_scale,
            vqvae=vqvae,
            vae_ckpt=vae_ckpt,
            decode_vq_to_image=decode_vq_to_image,
        )
        saved.append(
            {
                "idx": int(idx),
                "phase": phase,
                "name": name,
                "step": int(step),
                "mask_tokens": int(mask_count),
                "preview": str(path),
            }
        )
    return saved


def infer_one_sample(
    *,
    args,
    sample_dir: Path,
    output_dir: Path,
    data_root: Path | None,
    manifest_instructions: dict[str, str],
    empty_manifest_keys: set[str],
    tokenizer,
    model,
    model_device,
    vqvae,
    vae_scale: int,
    special_tokens: dict,
    decode_vq_to_image,
):
    import numpy as np
    import torch
    from generators.temporal_generator import generate_temporal_mask_then_unmask_from_original_image

    before_path = sample_dir / "before_ids.npy"
    if not before_path.is_file():
        raise FileNotFoundError(f"missing before_ids file: {before_path}")

    before_ids = np.load(before_path, allow_pickle=False)
    if before_ids.ndim != 2:
        raise ValueError(f"before_ids must be 2D, got shape {before_ids.shape}")
    prompt_text, prompt_meta = resolve_prompt(
        args=args,
        sample_dir=sample_dir,
        data_root=data_root,
        manifest_instructions=manifest_instructions,
        empty_manifest_keys=empty_manifest_keys,
    )

    mask_token = int(special_tokens["mask_token"])
    newline_token = int(special_tokens["newline_token"])
    vq_offset = int(special_tokens["image_token_offset"])
    answer_begin = int(special_tokens["answer_start"])
    answer_end = int(special_tokens["answer_end"])
    boi_token = int(special_tokens["boi"])
    eoi_token = int(special_tokens["eoi"])

    img_tokens, h, w, seq_len = build_image_tokens_from_before(
        before_ids,
        vq_offset=vq_offset,
        newline_token=newline_token,
    )
    full_seq, image_start, image_end, cfg_tail_start = build_full_sequence(
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        instruction=prompt_text,
        img_tokens=img_tokens,
        answer_begin=answer_begin,
        boi_token=boi_token,
        eoi_token=eoi_token,
        answer_end=answer_end,
    )
    uncond_text_prefix = build_uncond_prefix(
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        uncond_prompt=args.uncond_prompt,
    )

    prompt_tensor = torch.tensor(full_seq, dtype=torch.long, device=model_device).unsqueeze(0)
    uncond_text_tensor = torch.tensor(uncond_text_prefix, dtype=torch.long, device=model_device).unsqueeze(0)
    content_positions = build_content_positions(image_start=image_start, h=h, w=w)
    content_full_mask = torch.zeros(1, len(full_seq), dtype=torch.bool, device=model_device)
    content_full_mask[0, content_positions] = True

    vq_tokens, mask_local, debug_info = generate_temporal_mask_then_unmask_from_original_image(
        model=model,
        prompt=prompt_tensor,
        content_full_mask=content_full_mask,
        image_start=image_start,
        image_end=image_end,
        seq_len=seq_len,
        mask_timesteps=int(args.mask_timesteps),
        unmask_timesteps=int(args.unmask_timesteps),
        mask_confidence_threshold=float(args.mask_confidence_threshold),
        unmask_confidence_threshold=float(args.unmask_confidence_threshold),
        unmask_decode_mode=str(args.unmask_decode_mode),
        unmask_temperature=float(args.temperature),
        final_unmask_flush=bool(args.final_unmask_flush),
        cfg_scale=float(args.cfg_scale),
        cfg_img=float(args.cfg_img),
        uncond_text_prefix=uncond_text_tensor,
        cfg_tail_start=cfg_tail_start,
        mask_token_id=mask_token,
        newline_id=newline_token,
        temporal_history_decay=float(args.temporal_history_decay),
        temporal_history_scale=float(args.temporal_history_scale),
        mask_temporal_history_scale=args.mask_temporal_history_scale,
        unmask_temporal_history_scale=args.unmask_temporal_history_scale,
        temporal_history_mode=str(args.temporal_history_mode),
        temporal_history_gate_min=float(args.temporal_history_gate_min),
        temporal_history_gate_max=float(args.temporal_history_gate_max),
        temporal_history_max_steps=int(args.temporal_history_max_steps),
        return_debug=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_grid = mask_local.numpy().astype(bool, copy=False).reshape(h, w)
    pred_indices = mask_grid_to_indices(pred_grid)
    generated_ids = (vq_tokens.detach().cpu().numpy().astype(np.int64).reshape(h, w) - int(vq_offset))

    np.save(output_dir / "pred_edit_indices.npy", pred_indices)
    np.save(output_dir / "pred_mask_grid.npy", pred_grid)
    np.save(output_dir / "generated_ids.npy", generated_ids)

    if args.save_mask_preview:
        save_mask_preview(
            before_ids=before_ids,
            pred_grid=pred_grid,
            output_path=output_dir / "pred_mask_preview.png",
            model_device=model_device,
            vq_offset=vq_offset,
            vae_scale=vae_scale,
            vqvae=vqvae,
            vae_ckpt=args.vae_ckpt,
            decode_vq_to_image=decode_vq_to_image,
        )

    generated_path = output_dir / "generated.png"
    decode_vq_to_image(
        vq_tokens,
        str(generated_path),
        vae_ckpt=args.vae_ckpt,
        image_height=h * vae_scale,
        image_width=w * vae_scale,
        vqvae=vqvae,
    ).save(generated_path)

    before_flat_tokens = torch.tensor(img_tokens, dtype=torch.long).view(1, h * (w + 1))
    before_flat_tokens = before_flat_tokens[before_flat_tokens != int(newline_token)].view(1, seq_len)
    step_debug_files = []
    if args.save_step_debug:
        step_debug_files = save_step_debug_images(
            debug_info=debug_info,
            output_dir=output_dir,
            before_flat_tokens=before_flat_tokens,
            h=h,
            w=w,
            mask_token=mask_token,
            newline_token=newline_token,
            model_device=model_device,
            vae_scale=vae_scale,
            vqvae=vqvae,
            vae_ckpt=args.vae_ckpt,
            decode_vq_to_image=decode_vq_to_image,
        )

    gt_info = {"enabled": bool(args.save_ground_truth)}
    if args.save_ground_truth:
        gt_grid, gt_path = load_mask_grid(sample_dir, args.gt_mask_field, h=h, w=w)
        gt_info.update(
            {
                "mask_field": str(args.gt_mask_field),
                "mask_path": str(gt_path),
                "mask_found": gt_grid is not None,
            }
        )
        if gt_grid is not None:
            gt_indices = mask_grid_to_indices(gt_grid)
            np.save(output_dir / "gt_mask_grid.npy", gt_grid)
            np.save(output_dir / "gt_edit_indices.npy", gt_indices)
            intersection = int((pred_grid & gt_grid).sum())
            pred_count = int(pred_grid.sum())
            gt_count = int(gt_grid.sum())
            union = int((pred_grid | gt_grid).sum())
            gt_info.update(
                {
                    "gt_mask_tokens": int(gt_indices.shape[0]),
                    "pred_gt_intersection": intersection,
                    "pred_gt_union": union,
                    "pred_gt_iou": float(intersection / union) if union else 1.0,
                    "pred_precision": float(intersection / pred_count) if pred_count else 0.0,
                    "pred_recall": float(intersection / gt_count) if gt_count else 0.0,
                }
            )
            if args.save_mask_preview:
                save_mask_preview(
                    before_ids=before_ids,
                    pred_grid=gt_grid,
                    output_path=output_dir / "gt_mask_preview.png",
                    model_device=model_device,
                    vq_offset=vq_offset,
                    vae_scale=vae_scale,
                    vqvae=vqvae,
                    vae_ckpt=args.vae_ckpt,
                    decode_vq_to_image=decode_vq_to_image,
                )

        after_path = sample_dir / "after_ids.npy"
        gt_info["after_path"] = str(after_path)
        gt_info["after_found"] = after_path.is_file()
        if after_path.is_file():
            after_ids = np.load(after_path, allow_pickle=False)
            if after_ids.ndim == 2 and tuple(after_ids.shape) == (h, w):
                save_vq_ids_image(
                    ids_hw=after_ids,
                    output_path=output_dir / "ground_truth.png",
                    model_device=model_device,
                    vq_offset=vq_offset,
                    vae_scale=vae_scale,
                    vqvae=vqvae,
                    vae_ckpt=args.vae_ckpt,
                    decode_vq_to_image=decode_vq_to_image,
                )
                gt_info["generated_token_match"] = float((generated_ids == after_ids).mean())
            else:
                gt_info["after_shape_warning"] = str(after_ids.shape)

    debug_json = dict(debug_info)
    debug_json.pop("states", None)
    meta = {
        "sample_dir": str(sample_dir),
        "prompt": prompt_text,
        "prompt_source": prompt_meta,
        "token_h": int(h),
        "token_w": int(w),
        "total_content_tokens": int(h * w),
        "selected_tokens": int(pred_indices.shape[0]),
        "selected_ratio": float(pred_indices.shape[0] / max(h * w, 1)),
        "phase_1_criterion": "p_mask > mask_confidence_threshold and p_mask > p_current_token",
        "phase_2_criterion": (
            "Lumina/MaskGIT: sample VQ for current MASK positions, then keep low-confidence "
            "tokens masked by cosine schedule"
            if str(args.unmask_decode_mode) == "lumina"
            else "p_best_vq > unmask_confidence_threshold and p_best_vq > p_mask"
        ),
        "unmask_decode_mode": str(args.unmask_decode_mode),
        "mask_timesteps": int(args.mask_timesteps),
        "unmask_timesteps": int(args.unmask_timesteps),
        "mask_confidence_threshold": float(args.mask_confidence_threshold),
        "unmask_confidence_threshold": float(args.unmask_confidence_threshold),
        "temperature": float(args.temperature),
        "final_unmask_flush": bool(args.final_unmask_flush),
        "cfg_scale": float(args.cfg_scale),
        "cfg_img": float(args.cfg_img),
        "temporal_history_decay": float(args.temporal_history_decay),
        "temporal_history_scale": float(args.temporal_history_scale),
        "mask_temporal_history_scale": (
            float(args.mask_temporal_history_scale)
            if args.mask_temporal_history_scale is not None
            else float(args.temporal_history_scale)
        ),
        "unmask_temporal_history_scale": (
            float(args.unmask_temporal_history_scale)
            if args.unmask_temporal_history_scale is not None
            else float(args.temporal_history_scale)
        ),
        "temporal_history_mode": str(args.temporal_history_mode),
        "temporal_history_gate_min": float(args.temporal_history_gate_min),
        "temporal_history_gate_max": float(args.temporal_history_gate_max),
        "temporal_history_max_steps": int(args.temporal_history_max_steps),
        "generated_png": str(generated_path),
        "gt": gt_info,
        "step_debug_files": step_debug_files,
        "debug": debug_json,
    }
    if args.save_debug_json:
        (output_dir / "two_stage_debug.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(
        f"[ok] sample={sample_dir.name} mask={pred_indices.shape[0]}/{h * w} "
        f"generated={generated_path}",
        flush=True,
    )


def main():
    args = parse_args()

    import torch
    from diffusers import VQModel
    from transformers import AutoTokenizer

    from config import SPECIAL_TOKENS
    from model.modeling_temporal import TemporalTrainingModel
    from utils.generation_utils import setup_seed
    from utils.image_utils import decode_vq_to_image

    if args.seed:
        setup_seed(int(args.seed))

    if args.sample_list:
        sample_pairs = _read_sample_list(args.sample_list)
    elif args.sample_dir:
        sample_pairs = [(Path(args.sample_dir), Path(args.output_dir))]
    else:
        raise ValueError("Provide either --sample-dir or --sample-list")
    if not sample_pairs:
        print("[warn] empty sample list")
        return

    data_root = Path(args.data_root).resolve() if args.data_root else None
    manifest_instructions: dict[str, str] = {}
    empty_manifest_keys: set[str] = set()
    manifest_stats: dict = {}
    if args.instruction_source == "manifest":
        if not args.instruction_manifest:
            raise ValueError("--instruction-manifest is required when --instruction-source=manifest")
        manifest_path = Path(args.instruction_manifest).resolve()
        manifest_instructions, empty_manifest_keys, manifest_stats = load_instruction_manifest(
            manifest_path,
            args.instruction_field,
        )
        args.instruction_manifest = str(manifest_path)
        print(
            f"[info] loaded instruction manifest={manifest_path} "
            f"field={args.instruction_field} stats={manifest_stats}",
            flush=True,
        )

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    tokenizer_path = resolve_tokenizer_path(args.checkpoint, args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = TemporalTrainingModel.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "auto" else None,
    )
    if args.device != "auto":
        model = model.to(device)
    model.eval()
    model_device = next(model.parameters()).device

    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(model_device)
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)

    for idx, (sample_dir, output_dir) in enumerate(sample_pairs, start=1):
        print(f"[{idx}/{len(sample_pairs)}] sample={sample_dir} output={output_dir}", flush=True)
        infer_one_sample(
            args=args,
            sample_dir=sample_dir,
            output_dir=output_dir,
            data_root=data_root,
            manifest_instructions=manifest_instructions,
            empty_manifest_keys=empty_manifest_keys,
            tokenizer=tokenizer,
            model=model,
            model_device=model_device,
            vqvae=vqvae,
            vae_scale=vae_scale,
            special_tokens=SPECIAL_TOKENS,
            decode_vq_to_image=decode_vq_to_image,
        )


if __name__ == "__main__":
    main()
