
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from utils.generation_utils import cosine_schedule, gumbel_max_sample, mask_by_random_topk


def _get_wte(model):
    model_root = model.module if hasattr(model, "module") else model
    return model_root.model.transformer.wte


def _get_history_output_norm(model):
    model_root = model.module if hasattr(model, "module") else model
    return getattr(model_root, "history_output_norm", None)


def _get_history_gate_logit(model):
    model_root = model.module if hasattr(model, "module") else model
    return getattr(model_root, "history_gate_logit", None)


def rms_normalize_temporal(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (x.float() / rms).to(dtype=x.dtype)


def apply_temporal_rope(embed: torch.Tensor, step_id: int) -> torch.Tensor:
    if step_id <= 0:
        return embed

    hidden_dim = embed.shape[-1]
    half_dim = hidden_dim // 2
    if half_dim == 0:
        return embed

    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(half_dim, device=embed.device, dtype=torch.float32)
            / float(half_dim)
        )
    )
    theta = float(step_id) * inv_freq
    cos = torch.cos(theta).to(dtype=embed.dtype).view(*([1] * (embed.ndim - 1)), -1)
    sin = torch.sin(theta).to(dtype=embed.dtype).view(*([1] * (embed.ndim - 1)), -1)

    x1 = embed[..., :half_dim]
    x2 = embed[..., half_dim : 2 * half_dim]
    rotated = torch.cat(
        [
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ],
        dim=-1,
    )
    if hidden_dim % 2 == 0:
        return rotated
    return torch.cat([rotated, embed[..., -1:]], dim=-1)


def build_history_aware_embeddings(
    *,
    model,
    current_ids: torch.LongTensor,
    history_ids: Sequence[torch.LongTensor],
    history_token_mask: torch.BoolTensor,
    temporal_history_decay: float = 0.0,
    temporal_history_scale: float = 1.0,
    temporal_history_mode: str = "post_rms",
    temporal_history_gate_min: float = 0.002,
    temporal_history_gate_max: float = 0.04,
) -> torch.Tensor:
    wte = _get_wte(model)
    current_embed = wte(current_ids)
    if float(temporal_history_scale) == 0.0:
        return current_embed
    if not history_ids:
        return rms_normalize_temporal(current_embed) if temporal_history_mode in {"post_rms", "gated_post_rms"} else current_embed
    if temporal_history_mode not in {"post_rms", "gated_post_rms", "residual_norm", "gated_residual"}:
        raise ValueError(f"Unsupported temporal_history_mode: {temporal_history_mode}")

    frames = torch.stack([hist.to(current_ids.device) for hist in history_ids], dim=0)
    hist_embed = wte(frames)
    rotated_frames = []
    for frame_idx, frame_embed in enumerate(hist_embed, start=1):
        rotated = apply_temporal_rope(frame_embed, frame_idx)
        if temporal_history_decay > 0.0:
            decay = math.exp(-float(temporal_history_decay) * max(frame_idx - 1, 0))
            rotated = rotated * float(decay)
        rotated_frames.append(rotated)

    residual = torch.stack(rotated_frames, dim=0).sum(dim=0)
    residual = residual * history_token_mask.to(dtype=residual.dtype, device=residual.device).unsqueeze(-1)
    if temporal_history_mode in {"post_rms", "gated_post_rms"}:
        if temporal_history_mode == "gated_post_rms":
            gate_logit = _get_history_gate_logit(model)
            if gate_logit is None:
                gate = float(temporal_history_gate_max)
            else:
                gate = float(temporal_history_gate_min) + (
                    float(temporal_history_gate_max) - float(temporal_history_gate_min)
                ) * torch.sigmoid(gate_logit.float())
                gate = gate.to(device=residual.device, dtype=residual.dtype)
            residual = residual * gate
        return rms_normalize_temporal(current_embed + (residual * float(temporal_history_scale)).unsqueeze(0))
    norm = _get_history_output_norm(model)
    if norm is not None:
        norm_dtype = norm.weight.dtype if norm.weight is not None else residual.dtype
        residual = norm(residual.to(norm_dtype)).to(dtype=current_embed.dtype)
    else:
        residual = F.layer_norm(
            residual.float(),
            (residual.shape[-1],),
            eps=1e-5,
        ).to(dtype=current_embed.dtype)
    if temporal_history_mode == "gated_residual":
        gate_logit = _get_history_gate_logit(model)
        if gate_logit is None:
            gate = float(temporal_history_gate_max)
        else:
            gate = float(temporal_history_gate_min) + (
                float(temporal_history_gate_max) - float(temporal_history_gate_min)
            ) * torch.sigmoid(gate_logit.float())
            gate = gate.to(device=residual.device, dtype=residual.dtype)
        residual = residual * gate
    residual = residual * float(temporal_history_scale)
    return current_embed + residual.unsqueeze(0)


def build_temporal_cfg_embeddings(
    *,
    accumulated,
    uncond_text_prefix_embed,
    cfg_tail_start: int,
    content_positions: Sequence[int],
    uncond_img_mask_embed,
):
    cond_embed = accumulated
    tail_embed = accumulated[:, cfg_tail_start:, :]
    uncond_text_embed = torch.cat([uncond_text_prefix_embed, tail_embed], dim=1)

    uncond_img_embed = accumulated.clone()
    if content_positions:
        uncond_img_embed[:, content_positions, :] = uncond_img_mask_embed
    return cond_embed, uncond_text_embed, uncond_img_embed


@torch.no_grad()
def generate_temporal_mask_then_unmask_from_original_image(
    model,
    prompt: torch.LongTensor,
    *,
    content_full_mask: torch.BoolTensor,
    image_start: int,
    image_end: int,
    seq_len: int,
    mask_timesteps: int = 8,
    unmask_timesteps: int = 8,
    mask_confidence_threshold: float = 0.9,
    unmask_confidence_threshold: float = 0.9,
    unmask_decode_mode: str = "lumina",
    unmask_temperature: float = 1.0,
    final_unmask_flush: bool = True,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    cfg_scale: float = 0.0,
    cfg_img: float = 0.0,
    uncond_text_prefix: torch.LongTensor,
    cfg_tail_start: Optional[int] = None,
    codebook_size: int = 8192,
    text_vocab_size: Optional[int] = None,
    temporal_history_decay: float = 0.2,
    temporal_history_scale: float = 1.0,
    mask_temporal_history_scale: Optional[float] = None,
    unmask_temporal_history_scale: Optional[float] = None,
    temporal_history_mode: str = "post_rms",
    temporal_history_gate_min: float = 0.002,
    temporal_history_gate_max: float = 0.04,
    temporal_history_max_steps: Optional[int] = 12,
    initial_history_ids: Optional[Sequence[torch.LongTensor]] = None,
    generator: Optional[torch.Generator] = None,
    return_debug: bool = False,
):
    device = next(model.parameters()).device
    x = prompt.to(device).clone()
    content_full_mask = content_full_mask.to(device)
    debug_records: list[dict] = []
    debug_states: list[torch.LongTensor] = []

    if cfg_tail_start is None:
        cfg_tail_start = int(image_start - 1)

    if uncond_text_prefix.ndim == 1:
        uncond_text_prefix = uncond_text_prefix.unsqueeze(0)
    uncond_text_prefix = uncond_text_prefix.to(device)

    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = int(vocab_total - codebook_size)
    vocab_offset = int(text_vocab_size)
    vocab_end = vocab_offset + int(codebook_size)
    mask_id_int = int(mask_token_id)

    content_positions = content_full_mask[0].nonzero(as_tuple=False).squeeze(-1)
    history_token_mask = content_full_mask[0]
    history_states: list[torch.LongTensor] = []
    base_history_scale = float(temporal_history_scale)
    mask_history_scale = (
        base_history_scale if mask_temporal_history_scale is None else float(mask_temporal_history_scale)
    )
    unmask_history_scale = (
        base_history_scale if unmask_temporal_history_scale is None else float(unmask_temporal_history_scale)
    )
    if initial_history_ids:
        for hist in initial_history_ids:
            hist_tensor = hist.to(device)
            if hist_tensor.ndim == 2:
                if hist_tensor.size(0) != 1:
                    raise ValueError(f"initial history batch must be 1, got shape={tuple(hist_tensor.shape)}")
                hist_tensor = hist_tensor.squeeze(0)
            if hist_tensor.ndim != 1:
                raise ValueError(f"initial history state must be 1D or [1, L], got shape={tuple(hist_tensor.shape)}")
            if hist_tensor.numel() != x.size(1):
                raise ValueError(
                    f"initial history length {hist_tensor.numel()} does not match current length {x.size(1)}"
                )
            history_states.append(hist_tensor.detach().clone())
        if temporal_history_max_steps is not None:
            del history_states[max(0, int(temporal_history_max_steps)) :]
    wte = _get_wte(model)
    uncond_text_prefix_embed = wte(uncond_text_prefix)
    mask_token_vec = torch.full(
        (int(content_positions.numel()),),
        mask_id_int,
        dtype=torch.long,
        device=device,
    )
    uncond_img_mask_embed = wte(mask_token_vec).unsqueeze(0)

    def _slice_mask_plus_vq(full_logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        content_logits = full_logits[:, pos_mask, :]
        mask_part = content_logits[..., mask_id_int : mask_id_int + 1]
        vq_part = content_logits[..., vocab_offset:vocab_end]
        return torch.cat([mask_part, vq_part], dim=-1)

    def _snapshot_state(name: str, step: int, phase: str, **record):
        if not return_debug:
            return
        image_tokens = x[0, image_start:image_end].detach().cpu().clone()
        debug_states.append(image_tokens)
        content_now = x[0, content_positions] if content_positions.numel() > 0 else x.new_empty(0)
        debug_records.append(
            {
                "name": name,
                "phase": str(phase),
                "step": int(step),
                "mask_count": int((content_now == mask_id_int).sum().item()) if content_now.numel() else 0,
                "history_len": int(len(history_states)),
                **record,
            }
        )

    def _merged_logits(history_scale: float) -> torch.Tensor:
        history_aware_embed = build_history_aware_embeddings(
            model=model,
            current_ids=x,
            history_ids=history_states,
            history_token_mask=history_token_mask,
            temporal_history_decay=float(temporal_history_decay),
            temporal_history_scale=float(history_scale),
            temporal_history_mode=str(temporal_history_mode),
            temporal_history_gate_min=float(temporal_history_gate_min),
            temporal_history_gate_max=float(temporal_history_gate_max),
        )
        if cfg_scale > 0 or cfg_img > 0:
            cond_embed, uncond_text_embed, uncond_img_embed = build_temporal_cfg_embeddings(
                accumulated=history_aware_embed,
                uncond_text_prefix_embed=uncond_text_prefix_embed,
                cfg_tail_start=int(cfg_tail_start),
                content_positions=content_positions.tolist(),
                uncond_img_mask_embed=uncond_img_mask_embed,
            )
            uncond_text_mask = torch.cat(
                [
                    torch.zeros(1, uncond_text_prefix.size(1), dtype=torch.bool, device=device),
                    content_full_mask[:, cfg_tail_start:],
                ],
                dim=1,
            )
            cond_logits = _slice_mask_plus_vq(
                model(input_embeddings=cond_embed).logits, content_full_mask[0]
            )
            utext_logits = _slice_mask_plus_vq(
                model(input_embeddings=uncond_text_embed).logits, uncond_text_mask[0]
            )
            merged = cond_logits + cfg_scale * (cond_logits - utext_logits)
            if cfg_img > 0:
                uimg_logits = _slice_mask_plus_vq(
                    model(input_embeddings=uncond_img_embed).logits, content_full_mask[0]
                )
                merged = merged + cfg_img * (cond_logits - uimg_logits)
            return merged
        return _slice_mask_plus_vq(
            model(input_embeddings=history_aware_embed).logits, content_full_mask[0]
        )

    def _push_history(prev_x: torch.LongTensor):
        history_states.insert(0, prev_x.squeeze(0))
        if temporal_history_max_steps is not None:
            del history_states[max(0, int(temporal_history_max_steps)) :]

    mask_timesteps = max(0, int(mask_timesteps))
    unmask_timesteps = max(0, int(unmask_timesteps))
    mask_threshold = float(mask_confidence_threshold)
    unmask_threshold = float(unmask_confidence_threshold)
    unmask_decode_mode = str(unmask_decode_mode).replace("_", "-")
    if unmask_decode_mode not in {"lumina", "threshold"}:
        raise ValueError(f"Unsupported unmask_decode_mode: {unmask_decode_mode}")
    unmask_temperature = float(unmask_temperature)

    _snapshot_state(
        "initial",
        -1,
        "initial",
        commit_count=0,
        stopped=False,
        mask_confidence_threshold=mask_threshold,
        unmask_confidence_threshold=unmask_threshold,
        unmask_decode_mode=unmask_decode_mode,
        unmask_temperature=unmask_temperature,
        mask_temporal_history_scale=mask_history_scale,
        unmask_temporal_history_scale=unmask_history_scale,
    )

    for step in range(mask_timesteps):
        merged = _merged_logits(mask_history_scale)
        prev_x = x.detach().clone()
        current_at_content = x[0, content_positions]
        is_mask = current_at_content == mask_id_int
        probs = torch.softmax(merged[0].float(), dim=-1)
        mask_prob = probs[:, 0]
        current_vq_idx = (current_at_content - int(vocab_offset)).clamp(0, int(codebook_size) - 1)
        current_prob = probs[:, 1:].gather(1, current_vq_idx.unsqueeze(1)).squeeze(1)
        select = (~is_mask) & (mask_prob > mask_threshold) & (mask_prob > current_prob)
        commit_count = int(select.sum().item())
        if commit_count > 0:
            x[0, content_positions[select]] = mask_id_int

        has_unmasked = bool((~is_mask).any().item())
        _snapshot_state(
            "step",
            step,
            "mask",
            commit_count=commit_count,
            stopped=commit_count == 0,
            max_mask_confidence=float(mask_prob.max().item()) if mask_prob.numel() else None,
            mean_mask_confidence=float(mask_prob.mean().item()) if mask_prob.numel() else None,
            max_current_confidence=float(current_prob[~is_mask].max().item()) if has_unmasked else None,
            mean_current_confidence=float(current_prob[~is_mask].mean().item()) if has_unmasked else None,
            max_mask_minus_current=float((mask_prob - current_prob)[~is_mask].max().item()) if has_unmasked else None,
            temporal_history_scale=mask_history_scale,
        )
        if commit_count == 0:
            break
        _push_history(prev_x)

    mask_phase_content = x[0, content_positions] if content_positions.numel() else x.new_empty(0)
    pred_mask_local = (mask_phase_content == mask_id_int).detach().cpu()
    _snapshot_state(
        "phase_boundary",
        mask_timesteps,
        "boundary",
        commit_count=0,
        stopped=False,
        predicted_mask_tokens=int(pred_mask_local.sum().item()),
    )

    lumina_initial_mask_count = int(pred_mask_local.sum().item())
    for step in range(unmask_timesteps):
        merged = _merged_logits(unmask_history_scale)
        prev_x = x.detach().clone()
        current_at_content = x[0, content_positions]
        is_mask = current_at_content == mask_id_int
        mask_count_before = int(is_mask.sum().item())
        if mask_count_before == 0:
            _snapshot_state(
                "step",
                step,
                "unmask",
                commit_count=0,
                stopped=True,
                remaining_mask_tokens=0,
                temporal_history_scale=unmask_history_scale,
            )
            break

        if unmask_decode_mode == "lumina":
            if step < unmask_timesteps - 1:
                frac = cosine_schedule(
                    torch.tensor([(step + 1) / float(max(unmask_timesteps, 1))], device=device)
                )
                keep_n = int((float(lumina_initial_mask_count) * float(frac.item())) // 1)
                keep_n = max(1, keep_n)
            else:
                keep_n = 0

            local_logits = merged[0]
            vq_logits = local_logits[:, 1:]
            masked_local_idx = is_mask.nonzero(as_tuple=False).squeeze(1)
            masked_vq_logits = vq_logits[masked_local_idx]
            sampled_vq_idx = gumbel_max_sample(
                masked_vq_logits,
                unmask_temperature,
                generator=generator,
            )
            sampled_vq_tokens = sampled_vq_idx.to(dtype=x.dtype) + int(vocab_offset)
            probs = torch.softmax(masked_vq_logits.float(), dim=-1)
            confidence = probs.gather(1, sampled_vq_idx.unsqueeze(1)).squeeze(1)

            x[0, content_positions[masked_local_idx]] = sampled_vq_tokens
            keep_count = 0
            if keep_n > 0 and confidence.numel() > 0:
                keep_mask = mask_by_random_topk(
                    torch.tensor([keep_n], dtype=torch.long, device=device),
                    confidence.unsqueeze(0),
                    temperature=unmask_temperature,
                    generator=generator,
                ).squeeze(0)
                if bool(keep_mask.any().item()):
                    x[0, content_positions[masked_local_idx[keep_mask]]] = mask_id_int
                keep_count = int((x[0, content_positions] == mask_id_int).sum().item())

            remaining_mask_tokens = int((x[0, content_positions] == mask_id_int).sum().item())
            commit_count = int(mask_count_before - remaining_mask_tokens)
            _snapshot_state(
                "step",
                step,
                "unmask",
                commit_count=commit_count,
                stopped=False,
                decode_mode=unmask_decode_mode,
                sampled_count=int(mask_count_before),
                keep_n=int(keep_n),
                keep_count=int(keep_count),
                remaining_mask_tokens=remaining_mask_tokens,
                lumina_initial_mask_count=int(lumina_initial_mask_count),
                max_vq_confidence=float(confidence.max().item()) if confidence.numel() else None,
                mean_vq_confidence=float(confidence.mean().item()) if confidence.numel() else None,
                temporal_history_scale=unmask_history_scale,
            )
        else:
            probs = torch.softmax(merged[0].float(), dim=-1)
            mask_prob = probs[:, 0]
            best_vq_prob, best_vq_idx = probs[:, 1:].max(dim=-1)
            best_vq_tokens = best_vq_idx.to(dtype=x.dtype) + int(vocab_offset)
            select = is_mask & (best_vq_prob > unmask_threshold) & (best_vq_prob > mask_prob)
            commit_count = int(select.sum().item())
            if commit_count > 0:
                x[0, content_positions[select]] = best_vq_tokens[select]

            _snapshot_state(
                "step",
                step,
                "unmask",
                commit_count=commit_count,
                stopped=commit_count == 0,
                decode_mode=unmask_decode_mode,
                remaining_mask_tokens=int((x[0, content_positions] == mask_id_int).sum().item()),
                max_vq_confidence=float(best_vq_prob[is_mask].max().item()) if bool(is_mask.any().item()) else None,
                mean_vq_confidence=float(best_vq_prob[is_mask].mean().item()) if bool(is_mask.any().item()) else None,
                max_mask_confidence=float(mask_prob[is_mask].max().item()) if bool(is_mask.any().item()) else None,
                mean_mask_confidence=float(mask_prob[is_mask].mean().item()) if bool(is_mask.any().item()) else None,
                max_vq_minus_mask=float((best_vq_prob - mask_prob)[is_mask].max().item()) if bool(is_mask.any().item()) else None,
                temporal_history_scale=unmask_history_scale,
            )
        if commit_count == 0:
            if unmask_decode_mode == "threshold":
                break
            continue
        _push_history(prev_x)

    remaining_mask = (x[0] == mask_id_int) & content_full_mask[0]
    final_mask_count_before_flush = int(remaining_mask.sum().item())
    if final_unmask_flush and bool(remaining_mask.any().item()):
        prev_x = x.detach().clone()
        merged = _merged_logits(unmask_history_scale)
        vq_logits = merged[0, :, 1:]
        remaining_local = (x[0, content_positions] == mask_id_int)
        remaining_vq = vq_logits[remaining_local]
        remaining_argmax = remaining_vq.argmax(dim=-1) + int(vocab_offset)
        x[0, content_positions[remaining_local]] = remaining_argmax.to(x.dtype)
        _snapshot_state(
            "final_flush",
            unmask_timesteps,
            "unmask",
            commit_count=final_mask_count_before_flush,
            stopped=False,
            final_mask_count_before_flush=final_mask_count_before_flush,
            remaining_mask_tokens=0,
            temporal_history_scale=unmask_history_scale,
        )
        if final_mask_count_before_flush > 0:
            _push_history(prev_x)

    vq_ids = x[0, image_start:image_end]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    if return_debug:
        return vq_ids, pred_mask_local, {
            "records": debug_records,
            "states": debug_states,
            "selected_tokens": int(pred_mask_local.sum().item()),
            "total_content_tokens": int(content_positions.numel()),
            "mask_confidence_threshold": mask_threshold,
            "unmask_confidence_threshold": unmask_threshold,
            "unmask_decode_mode": unmask_decode_mode,
            "unmask_temperature": unmask_temperature,
            "temporal_history_scale": base_history_scale,
            "mask_temporal_history_scale": mask_history_scale,
            "unmask_temporal_history_scale": unmask_history_scale,
            "final_mask_count_before_flush": final_mask_count_before_flush,
            "final_unmask_flush": bool(final_unmask_flush),
        }
    return vq_ids, pred_mask_local
