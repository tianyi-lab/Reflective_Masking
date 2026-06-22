
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from config import SPECIAL_TOKENS

from .modeling_llada import LLaDAModelLM
from .modeling_xllmx_dimoo import LLaDAForMultiModalGeneration


DEFAULT_MASK_TOKEN_ID = int(SPECIAL_TOKENS["mask_token"])
_BOI = int(SPECIAL_TOKENS["boi"])
_EOI = int(SPECIAL_TOKENS["eoi"])
_ANS_BEGIN = int(SPECIAL_TOKENS["answer_start"])
_ANS_END = int(SPECIAL_TOKENS["answer_end"])


def _create_attention_mask(original_lengths: list[int], max_tokens: int, device) -> torch.Tensor:
    batch_size = len(original_lengths)
    attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=device)
    for row_idx, length in enumerate(original_lengths):
        attention_mask[row_idx, :length] = 1
    return attention_mask


def _pad_nested_int_lists(
    sequences: Sequence[Sequence[int]],
    max_tokens: int,
    *,
    pad_value: int,
    device,
) -> torch.Tensor:
    import numpy as _np
    arr = _np.full((len(sequences), max_tokens), int(pad_value), dtype=_np.int64)
    for i, seq in enumerate(sequences):
        arr[i, : len(seq)] = seq
    return torch.from_numpy(arr).to(device, non_blocking=True)


def _pad_nested_float_lists(
    sequences: Sequence[Sequence[float]],
    max_tokens: int,
    *,
    pad_value: float,
    device,
) -> torch.Tensor:
    import numpy as _np
    arr = _np.full((len(sequences), max_tokens), float(pad_value), dtype=_np.float32)
    for i, seq in enumerate(sequences):
        arr[i, : len(seq)] = seq
    return torch.from_numpy(arr).to(device, non_blocking=True)


def _chunked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor | None = None,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    flat_weights = None if loss_weights is None else loss_weights.reshape(-1).to(dtype=torch.float32)
    total_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    normalizer = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, flat_logits.size(0), chunk_size):
        chunk_log = flat_logits[start : start + chunk_size]
        chunk_lbl = flat_labels[start : start + chunk_size]
        chunk_valid = chunk_lbl != ignore_index
        if flat_weights is None:
            if not bool(chunk_valid.any()):
                continue
            total_loss = total_loss + F.cross_entropy(
                chunk_log[chunk_valid].float(),
                chunk_lbl[chunk_valid],
                reduction="sum",
            )
            normalizer = normalizer + chunk_valid.sum().to(dtype=torch.float32)
        else:
            chunk_weight = flat_weights[start : start + chunk_size]
            chunk_valid = chunk_valid & (chunk_weight > 0)
            if not bool(chunk_valid.any()):
                continue
            chunk_loss = F.cross_entropy(
                chunk_log[chunk_valid].float(),
                chunk_lbl[chunk_valid],
                reduction="none",
            )
            selected_weight = chunk_weight[chunk_valid]
            total_loss = total_loss + (chunk_loss * selected_weight).sum()
            normalizer = normalizer + selected_weight.sum()
    if float(normalizer.item()) <= 0.0:
        return flat_logits.new_zeros((), requires_grad=True)
    return total_loss / normalizer


def _finite_tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, float | int | str]:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    stats: dict[str, float | int | str] = {
        "name": name,
        "shape": "x".join(str(dim) for dim in detached.shape),
        "finite": int(finite.sum().item()),
        "total": int(detached.numel()),
    }
    if bool(finite.any()):
        finite_values = detached[finite].float()
        stats["min"] = float(finite_values.min().item())
        stats["max"] = float(finite_values.max().item())
        stats["mean"] = float(finite_values.mean().item())
    return stats


def _rms_normalize_temporal(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (x.float() / rms).to(dtype=x.dtype)


def _bounded_history_gate(
    model,
    *,
    gate_min: float,
    gate_max: float,
    device,
    dtype,
) -> torch.Tensor:
    gate_logit = getattr(model, "history_gate_logit", None)
    if gate_logit is None:
        return torch.tensor(float(gate_max), device=device, dtype=dtype)
    gate_min = float(gate_min)
    gate_max = float(gate_max)
    gate = gate_min + (gate_max - gate_min) * torch.sigmoid(gate_logit.float())
    return gate.to(device=device, dtype=dtype)


def _record_last_history_gate(model, gate: torch.Tensor | None) -> None:
    if gate is None:
        model._last_temporal_history_gate = None
    else:
        model._last_temporal_history_gate = gate.detach().float().mean()


def _postprocess_history_img(history_img_list, editable_indices, mask_token):
    if len(history_img_list) <= 1:
        return
    for idx in editable_indices:
        for hist_idx in range(1, len(history_img_list)):
            prev_tok = int(history_img_list[hist_idx - 1][idx])
            curr_tok = int(history_img_list[hist_idx][idx])
            if prev_tok != curr_tok and prev_tok != int(mask_token) and curr_tok != int(mask_token):
                history_img_list[hist_idx - 1][idx] = int(mask_token)


def _wrap_img_as_full_seq(inst_tokens, img_tokens):
    return list(inst_tokens) + [_ANS_BEGIN, _BOI] + list(img_tokens) + [_EOI, _ANS_END]


def _history_to_padded_numpy(history, max_tokens: int) -> np.ndarray:
    if isinstance(history, np.ndarray):
        arr = history
    else:
        arr = np.asarray(history)
    if arr.dtype != np.int64:
        arr = arr.astype(np.int64, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"history must be 2D (T+1, seq_len), got shape {arr.shape}")
    t_plus_1, seq_len = arr.shape
    if seq_len > max_tokens:
        raise ValueError(f"history seq_len {seq_len} exceeds max_tokens {max_tokens}")
    if seq_len < max_tokens:
        arr = np.pad(arr, ((0, 0), (0, max_tokens - seq_len)), mode="constant", constant_values=0)
    return np.ascontiguousarray(arr)


def _apply_temporal_rope_batched(
    emb: torch.Tensor,
    step_ids: torch.Tensor,
) -> torch.Tensor:
    if emb.shape[0] == 0:
        return emb
    hidden_dim = emb.shape[-1]
    half_dim = hidden_dim // 2
    if half_dim == 0:
        return emb

    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(half_dim, device=emb.device, dtype=torch.float32)
            / float(half_dim)
        )
    )
    theta = step_ids.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = torch.cos(theta).to(dtype=emb.dtype).unsqueeze(1)
    sin = torch.sin(theta).to(dtype=emb.dtype).unsqueeze(1)

    x1 = emb[..., :half_dim]
    x2 = emb[..., half_dim : 2 * half_dim]
    rotated = torch.cat(
        [
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ],
        dim=-1,
    )
    if hidden_dim % 2 == 0:
        return rotated
    return torch.cat([rotated, emb[..., -1:]], dim=-1)


def _build_temporal_input_embeddings(
    model,
    current_input_ids_list: Sequence[Sequence[int]],
    history_token_ids_list: Sequence[Sequence[Sequence[int]]],
    history_distance_lists: Sequence[Sequence[int]],
    history_token_mask_list: Sequence[Sequence[bool]] | None = None,
    *,
    max_tokens: int,
    device,
    temporal_history_decay: float = 0.0,
    temporal_history_scale: float = 1.0,
    temporal_history_mode: str = "post_rms",
    temporal_history_gate_min: float = 0.002,
    temporal_history_gate_max: float = 0.04,
) -> torch.Tensor:
    if len(current_input_ids_list) != len(history_token_ids_list):
        raise ValueError("current_input_ids_list and history_token_ids_list must match batch size")
    if len(history_token_ids_list) != len(history_distance_lists):
        raise ValueError("history_token_ids_list and history_distance_lists must match batch size")
    if history_token_mask_list is not None and len(history_token_mask_list) != len(current_input_ids_list):
        raise ValueError("history_token_mask_list batch size must match input_ids")
    wte = model.model.transformer.wte
    current_tensor = _pad_nested_int_lists(current_input_ids_list, max_tokens, pad_value=0, device=device)
    current_embeddings = wte(current_tensor)
    _record_last_history_gate(model, None)
    if float(temporal_history_scale) == 0.0:
        return current_embeddings
    if temporal_history_mode not in {"post_rms", "gated_post_rms", "residual_norm", "gated_residual"}:
        raise ValueError(f"Unsupported temporal_history_mode: {temporal_history_mode}")

    accumulated_embeddings = []
    if history_token_mask_list is None:
        history_token_mask_list = [None] * len(current_input_ids_list)

    for current_emb, history, distances, token_mask in zip(
        current_embeddings,
        history_token_ids_list,
        history_distance_lists,
        history_token_mask_list,
    ):
        if len(history) != len(distances):
            raise ValueError("Each history sample must have the same number of frames and distances")

        if len(history) == 0:
            acc = current_emb
        else:
            frames_np = _history_to_padded_numpy(history, max_tokens)
            frames = torch.from_numpy(frames_np).to(device, non_blocking=True)
            hist_emb = wte(frames)
            step_ids = torch.tensor(list(distances), device=device, dtype=torch.float32)
            rotated = _apply_temporal_rope_batched(hist_emb, step_ids)

            if token_mask is not None:
                token_mask_tensor = _pad_nested_float_lists(
                    [token_mask],
                    max_tokens,
                    pad_value=0.0,
                    device=device,
                )[0].to(dtype=rotated.dtype)
                rotated = rotated * token_mask_tensor.unsqueeze(0).unsqueeze(-1)

            if temporal_history_decay > 0.0:
                decay_weights = torch.exp(
                    -float(temporal_history_decay)
                    * torch.clamp(step_ids - 1.0, min=0.0)
                ).to(device=device, dtype=rotated.dtype)
                rotated = rotated * decay_weights.view(-1, 1, 1)

            residual = rotated.sum(dim=0)
            if temporal_history_mode in {"post_rms", "gated_post_rms"}:
                if temporal_history_mode == "gated_post_rms":
                    gate = _bounded_history_gate(
                        model,
                        gate_min=float(temporal_history_gate_min),
                        gate_max=float(temporal_history_gate_max),
                        device=device,
                        dtype=residual.dtype,
                    )
                    _record_last_history_gate(model, gate)
                    residual = residual * gate
                acc = current_emb + residual * float(temporal_history_scale)
            else:
                if hasattr(model, "history_output_norm"):
                    norm = model.history_output_norm
                    norm_dtype = norm.weight.dtype if norm.weight is not None else residual.dtype
                    residual = norm(residual.to(norm_dtype)).to(dtype=rotated.dtype)
                else:
                    residual = F.layer_norm(
                        residual.float(),
                        (residual.shape[-1],),
                        eps=1e-5,
                    ).to(dtype=rotated.dtype)
                if temporal_history_mode == "gated_residual":
                    gate = _bounded_history_gate(
                        model,
                        gate_min=float(temporal_history_gate_min),
                        gate_max=float(temporal_history_gate_max),
                        device=device,
                        dtype=residual.dtype,
                    )
                    _record_last_history_gate(model, gate)
                    residual = residual * gate
                residual = residual * float(temporal_history_scale)
                acc = current_emb + residual

        if temporal_history_mode in {"post_rms", "gated_post_rms"}:
            acc = _rms_normalize_temporal(acc)

        accumulated_embeddings.append(acc)
    return torch.stack(accumulated_embeddings, dim=0)


class TemporalTrainingModel(LLaDAForMultiModalGeneration):
    all_tied_weights_keys: dict[str, str] = {}

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.all_tied_weights_keys = dict(getattr(self, "all_tied_weights_keys", {}) or {})
        hidden_size = int(getattr(config, "d_model", getattr(config, "hidden_size", 0)))
        if hidden_size <= 0:
            raise ValueError("TemporalTrainingModel requires config.d_model or config.hidden_size")
        self.history_output_norm = nn.LayerNorm(hidden_size)
        self.history_gate_logit = nn.Parameter(torch.zeros(1))

    def tie_weights(self, missing_keys=None, recompute_mapping: bool = True):
        del missing_keys, recompute_mapping
        return super().tie_weights()

    def forward(
        self,
        input_ids=None,
        labels=None,
        *,
        loss_weights=None,
        history_token_ids_list=None,
        history_distance_lists=None,
        history_token_mask_list=None,
        temporal_history_decay: float = 0.0,
        temporal_history_scale: float = 1.0,
        temporal_history_mode: str = "post_rms",
        temporal_history_gate_min: float = 0.002,
        temporal_history_gate_max: float = 0.04,
        debug_nonfinite: bool = False,
        input_embeddings=None,
        infer=False,
        use_cache=False,
        to_compute_mask=None,
        cat="",
        **kwargs,
    ):
        if input_embeddings is not None:
            return LLaDAForMultiModalGeneration.forward(
                self,
                input_embeddings=input_embeddings,
                use_cache=use_cache,
                to_compute_mask=to_compute_mask,
                cat=cat,
                **kwargs,
            )

        if infer:
            return super().forward(
                input_ids=input_ids,
                labels=labels,
                infer=True,
                use_cache=use_cache,
                to_compute_mask=to_compute_mask,
                cat=cat,
                **kwargs,
            )

        if input_ids is None:
            raise ValueError("input_ids is required")
        if labels is None:
            raise ValueError("labels is required")

        device = self.device
        if len(input_ids) != len(labels):
            raise ValueError("input_ids and labels must have matching batch size")
        max_tokens = max(len(seq) for seq in input_ids)
        input_tensor = _pad_nested_int_lists(input_ids, max_tokens, pad_value=0, device=device)
        label_tensor = _pad_nested_int_lists(labels, max_tokens, pad_value=-100, device=device)
        loss_weight_tensor = None
        if loss_weights is not None:
            loss_weight_tensor = _pad_nested_float_lists(
                loss_weights,
                max_tokens,
                pad_value=0.0,
                device=device,
            )
        attention_bias = None

        if history_token_ids_list is not None:
            if len(history_token_ids_list) != len(input_ids):
                raise ValueError("history_token_ids_list batch size must match input_ids")
            if history_distance_lists is None:
                raise ValueError("history_distance_lists is required when history_token_ids_list is provided")

            input_embeddings = _build_temporal_input_embeddings(
                self,
                input_ids,
                history_token_ids_list,
                history_distance_lists,
                history_token_mask_list=history_token_mask_list,
                max_tokens=max_tokens,
                device=device,
                temporal_history_decay=float(temporal_history_decay),
                temporal_history_scale=float(temporal_history_scale),
                temporal_history_mode=str(temporal_history_mode),
                temporal_history_gate_min=float(temporal_history_gate_min),
                temporal_history_gate_max=float(temporal_history_gate_max),
            )
            if debug_nonfinite and not torch.isfinite(input_embeddings).all():
                print(
                    "[temporal-debug] nonfinite input_embeddings",
                    _finite_tensor_stats("input_embeddings", input_embeddings),
                    flush=True,
                )
            output = LLaDAModelLM.forward(
                self,
                inputs_embeds=input_embeddings,
                attention_bias=attention_bias,
                use_cache=use_cache,
                to_compute_mask=to_compute_mask,
                cat=cat,
                **kwargs,
            )
        else:
            output = LLaDAModelLM.forward(
                self,
                input_ids=input_tensor,
                attention_bias=attention_bias,
                use_cache=use_cache,
                to_compute_mask=to_compute_mask,
                cat=cat,
                **kwargs,
            )

        logits = output.logits
        if debug_nonfinite and not torch.isfinite(logits).all():
            debug_stats = [_finite_tensor_stats("logits", logits)]
            supervised_mask = label_tensor != -100
            if loss_weight_tensor is not None:
                supervised_mask = supervised_mask & (loss_weight_tensor > 0)
            if bool(supervised_mask.any()):
                debug_stats.append(_finite_tensor_stats("supervised_logits", logits[supervised_mask]))
                debug_stats.append(_finite_tensor_stats("supervised_labels", label_tensor[supervised_mask]))
            print("[temporal-debug] nonfinite logits", debug_stats, flush=True)
        return _chunked_cross_entropy(
            logits,
            label_tensor,
            loss_weights=loss_weight_tensor,
            ignore_index=-100,
        )
