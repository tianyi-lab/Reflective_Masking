
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


def _coerce_positive_int(value, *, name: str) -> int:
    coerced = int(value)
    if coerced < 1:
        raise ValueError(f"{name} must be >= 1, got {coerced}")
    return coerced


def _get_steps_for_block(*, total_steps: int, num_blocks: int, block_idx: int) -> int:
    base_steps = total_steps // num_blocks
    extra_steps = total_steps % num_blocks
    return base_steps + int(block_idx < extra_steps)


def _apply_generation_step_updates(
    *,
    x: torch.Tensor,
    candidate_ids: torch.Tensor,
    transfer_index: torch.Tensor,
    remask_index: torch.Tensor,
    mask_id: int,
) -> bool:
    next_x = x.clone()
    next_x[remask_index] = mask_id
    next_x[transfer_index] = candidate_ids[transfer_index]
    changed = not torch.equal(next_x, x)
    if changed:
        x.copy_(next_x)
    return changed


def _build_generated_index(
    prompt_lens: list[int],
    max_new_tokens: int,
    total_len: int,
    device: torch.device,
) -> torch.Tensor:
    generated_index = torch.zeros(
        (len(prompt_lens), total_len), dtype=torch.bool, device=device
    )
    for batch_idx, prompt_len in enumerate(prompt_lens):
        valid_end = min(prompt_len + max_new_tokens, total_len)
        generated_index[batch_idx, prompt_len:valid_end] = True
    return generated_index


def _select_generation_step_updates(
    *,
    x: torch.Tensor,
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    generated_index: torch.Tensor,
    prompt_lens: list[int],
    max_new_tokens: int,
    total_len: int,
    block_idx: int,
    block_size: int,
    reveal_token_num_per_step: int,
    mask_id: int,
    remasking: str,
    temperature: float,
    allow_mask_token_candidate: bool,
    global_step_idx: int,
    remask_step_interval: int,
    remask_counts: torch.Tensor | None = None,
    max_remask_times: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    remask_step_interval = _coerce_positive_int(
        remask_step_interval, name="remask_step_interval"
    )
    reveal_token_num_per_step = _coerce_positive_int(
        reveal_token_num_per_step, name="reveal_token_num_per_step"
    )
    max_remask_times = int(max_remask_times)
    if max_remask_times < -1:
        raise ValueError(
            f"max_remask_times must be >= -1, got {max_remask_times}"
        )
    if max_remask_times > 0 and remask_counts is None:
        raise ValueError("remask_counts is required when max_remask_times > 0")

    p = F.softmax(logits, dim=-1)
    mask_p = p[:, :, mask_id]
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(predicted_ids, -1)), -1
    )
    candidate_ids = torch.where(mask_index, predicted_ids, x)
    current_token_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x, -1)), -1
    )
    transfer_index = torch.zeros_like(mask_index)
    remask_index = torch.zeros_like(mask_index)

    for batch_idx, prompt_len in enumerate(prompt_lens):
        block_end = min(
            prompt_len + (block_idx + 1) * block_size,
            prompt_len + max_new_tokens,
            total_len,
        )

        revealable_index = mask_index[batch_idx] & generated_index[batch_idx]
        revealable_index[block_end:] = False
        revealable_index &= predicted_ids[batch_idx] != mask_id
        if torch.any(revealable_index):
            reveal_scores = torch.where(
                revealable_index,
                predicted_p[batch_idx],
                torch.full_like(predicted_p[batch_idx], -torch.inf),
            )
            reveal_count = min(
                reveal_token_num_per_step,
                int(revealable_index.sum().item()),
            )
            _, selected_indices = torch.topk(reveal_scores, k=reveal_count)
            transfer_index[batch_idx, selected_indices] = True

        if global_step_idx % remask_step_interval != 0 or max_remask_times == 0:
            continue

        remaskable_index = (~mask_index[batch_idx]) & generated_index[batch_idx]
        remaskable_index[block_end:] = False
        if max_remask_times > 0:
            remaskable_index &= remask_counts[batch_idx] < max_remask_times
        remask_index[batch_idx] = remaskable_index & (
            mask_p[batch_idx] > current_token_p[batch_idx]
        )

    return candidate_ids, transfer_index, remask_index


@dataclass
class MDLMSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None
    )
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False
    allow_mask_token_candidate: bool = True
    remask_step_interval: int = 1
    reveal_token_num_per_step: int = 1
    max_remask_times: int = -1


@dataclass
class MDLMSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        if config is None:
            config = MDLMSamplerConfig()

        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        allow_mask_token_candidate = _coerce_bool(
            kwargs.get(
                "allow_mask_token_candidate", config.allow_mask_token_candidate
            )
        )
        remask_step_interval = _coerce_positive_int(
            kwargs.get("remask_step_interval", config.remask_step_interval),
            name="remask_step_interval",
        )
        reveal_token_num_per_step = _coerce_positive_int(
            kwargs.get(
                "reveal_token_num_per_step", config.reveal_token_num_per_step
            ),
            name="reveal_token_num_per_step",
        )
        max_remask_times = int(
            kwargs.get("max_remask_times", config.max_remask_times)
        )
        if max_remask_times < -1:
            raise ValueError(
                f"max_remask_times must be >= -1, got {max_remask_times}"
            )

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id
            )
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1
        generated_index = _build_generated_index(
            prompt_lens=prompt_lens,
            max_new_tokens=max_new_tokens,
            total_len=T,
            device=x.device,
        )

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        num_blocks = math.ceil(max_new_tokens / block_size)
        histories = [x.clone()] if return_dict else None
        global_step_idx = 0
        remask_counts = (
            torch.zeros_like(x, dtype=torch.int64)
            if max_remask_times >= 0
            else None
        )

        for b in range(num_blocks):
            steps_for_block = _get_steps_for_block(
                total_steps=steps,
                num_blocks=num_blocks,
                block_idx=b,
            )

            for _ in range(steps_for_block):
                mask_index = x == mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                if not allow_mask_token_candidate:
                    logits[:, :, mask_id] = -torch.inf

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                candidate_ids, transfer_index, remask_index = (
                    _select_generation_step_updates(
                        x=x,
                        logits=logits,
                        mask_index=mask_index,
                        generated_index=generated_index,
                        prompt_lens=prompt_lens,
                        max_new_tokens=max_new_tokens,
                        total_len=T,
                        block_idx=b,
                        block_size=block_size,
                        reveal_token_num_per_step=reveal_token_num_per_step,
                        mask_id=mask_id,
                        remasking=remasking,
                        temperature=temperature,
                        allow_mask_token_candidate=allow_mask_token_candidate,
                        global_step_idx=global_step_idx,
                        remask_step_interval=remask_step_interval,
                        remask_counts=remask_counts,
                        max_remask_times=max_remask_times,
                    )
                )

                if remask_counts is not None:
                    remask_counts += remask_index.to(remask_counts.dtype)
                changed = _apply_generation_step_updates(
                    x=x,
                    candidate_ids=candidate_ids,
                    transfer_index=transfer_index,
                    remask_index=remask_index,
                    mask_id=mask_id,
                )
                if not changed:
                    break
                if histories is not None:
                    histories.append(x.clone())
                global_step_idx += 1
            else:
                continue
            break

        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> BaseSamplerOutput | torch.Tensor:
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                for j in range(B):
                    end_j = start + widths[j]
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)
