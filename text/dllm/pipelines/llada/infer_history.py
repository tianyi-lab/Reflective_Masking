
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import dllm
import torch

from dllm.core.samplers import BaseSamplerOutput, MDLMSamplerConfig
from dllm.core.samplers.mdlm import (
    _apply_generation_step_updates,
    _build_generated_index,
    _coerce_positive_int,
    _get_steps_for_block,
    _select_generation_step_updates,
)
from dllm.pipelines.llada.models.history_wrapper import (
    LLaDASyntheticRevisionHistoryModel,
    load_llada_history_model,
    load_llada_history_tokenizer,
)


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


@dataclass
class LLaDATraceEvalSamplerConfig(MDLMSamplerConfig):

    max_new_tokens: int = 256
    steps: int = 256
    block_size: int = 256


@dataclass
class LLaDATraceSamplerOutput(BaseSamplerOutput):
    candidate_ids: list[torch.Tensor] | None = None
    transfer_indices: list[torch.Tensor] | None = None
    mask_indices: list[torch.Tensor] | None = None
    remask_indices: list[torch.Tensor] | None = None
    top1_token_ids: list[torch.Tensor] | None = None
    top1_confidences: list[torch.Tensor] | None = None

def _append_generation_history(
    history_buffer: list[torch.Tensor],
    previous_state: torch.Tensor,
    history_max_steps: int | None,
) -> list[torch.Tensor]:
    updated = [*history_buffer, previous_state]
    if history_max_steps is not None:
        updated = updated[-history_max_steps:]
    return updated


def _build_generation_history_tensors(
    history_buffer: list[torch.Tensor],
    *,
    history_token_mask: torch.Tensor,
    history_max_steps: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if history_max_steps is not None:
        history_buffer = history_buffer[-history_max_steps:]

    batch_size, total_len = history_token_mask.shape
    device = history_token_mask.device
    if not history_buffer:
        return (
            torch.empty(
                (batch_size, 0, total_len),
                dtype=torch.long,
                device=device,
            ),
            torch.empty((batch_size, 0), dtype=torch.long, device=device),
            torch.empty((batch_size, 0), dtype=torch.bool, device=device),
        )

    reversed_history = list(reversed(history_buffer))
    history_input_ids = torch.stack(reversed_history, dim=1)
    history_len = history_input_ids.shape[1]
    history_distances = torch.arange(
        1,
        history_len + 1,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(batch_size, -1)
    history_valid_mask = torch.ones(
        (batch_size, history_len),
        dtype=torch.bool,
        device=device,
    )
    return history_input_ids, history_distances, history_valid_mask


@dataclass
class LLaDAHistoryTraceEvalSamplerConfig(LLaDATraceEvalSamplerConfig):
    temporal_history_max_steps: int | None = None


@dataclass
class LLaDAHistoryTraceModelArguments(dllm.utils.ModelArguments):
    temporal_history_decay: float = 0.0
    temporal_history_max_steps: int | None = None
    temporal_distance_encoding: str = "rope"


@dataclass
class LLaDAHistoryTraceSampler:
    model: torch.nn.Module
    tokenizer: object

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> LLaDATraceSamplerOutput | torch.Tensor:
        if config is None:
            config = LLaDAHistoryTraceEvalSamplerConfig()

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
        return_top1 = _coerce_bool(kwargs.get("return_top1", False))
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
        history_max_steps = kwargs.get(
            "temporal_history_max_steps",
            getattr(config, "temporal_history_max_steps", None),
        )
        if history_max_steps is None:
            history_max_steps = getattr(
                getattr(self.model, "config", None),
                "temporal_history_max_steps",
                None,
            )
        if history_max_steps is not None:
            history_max_steps = _coerce_positive_int(
                history_max_steps,
                name="temporal_history_max_steps",
            )

        del stochastic_transfer

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(prompt, list) and len(prompt) == 0 else prompt
                for prompt in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(prompt, dtype=torch.long, device=self.model.device)
                for prompt in inputs
            ]
        prompt_lens = [prompt.shape[0] for prompt in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        batch_size = len(inputs)
        total_len = max_length

        x = torch.full(
            (batch_size, total_len),
            eos_id,
            dtype=torch.long,
            device=self.model.device,
        )
        for index, prompt in enumerate(inputs):
            x[index, : prompt_lens[index]] = prompt
            x[index, prompt_lens[index] : prompt_lens[index] + max_new_tokens] = mask_id
        attention_mask = torch.zeros(
            (batch_size, total_len),
            dtype=torch.long,
            device=self.model.device,
        )
        for index, prompt_len in enumerate(prompt_lens):
            valid_end = min(prompt_len + max_new_tokens, total_len)
            attention_mask[index, :valid_end] = 1
        generated_index = _build_generated_index(
            prompt_lens=prompt_lens,
            max_new_tokens=max_new_tokens,
            total_len=total_len,
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
        candidate_ids = [] if return_dict else None
        transfer_indices = [] if return_dict else None
        mask_indices = [] if return_dict else None
        remask_indices = [] if return_dict else None
        top1_token_ids = [] if return_dict and return_top1 else None
        top1_confidences = [] if return_dict and return_top1 else None
        temporal_history: list[torch.Tensor] = []
        global_step_idx = 0
        remask_counts = (
            torch.zeros_like(x, dtype=torch.int64)
            if max_remask_times >= 0
            else None
        )

        for block_index in range(num_blocks):
            steps_for_block = _get_steps_for_block(
                total_steps=steps,
                num_blocks=num_blocks,
                block_idx=block_index,
            )

            for _ in range(steps_for_block):
                mask_index = x == mask_id
                history_input_ids, history_distances, history_valid_mask = (
                    _build_generation_history_tensors(
                        temporal_history,
                        history_token_mask=generated_index,
                        history_max_steps=history_max_steps,
                    )
                )

                model_kwargs = {
                    "history_input_ids": history_input_ids,
                    "history_distances": history_distances,
                    "history_valid_mask": history_valid_mask,
                    "history_token_mask": generated_index,
                }

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_,
                        attention_mask=attention_mask.repeat(2, 1),
                        history_input_ids=history_input_ids.repeat(2, 1, 1),
                        history_distances=history_distances.repeat(2, 1),
                        history_valid_mask=history_valid_mask.repeat(2, 1),
                        history_token_mask=generated_index.repeat(2, 1),
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x,
                        attention_mask=attention_mask,
                        **model_kwargs,
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

                if top1_token_ids is not None and top1_confidences is not None:
                    step_top1_confidences, step_top1_token_ids = torch.max(
                        torch.softmax(logits, dim=-1),
                        dim=-1,
                    )

                step_candidate_ids, transfer_index, remask_index = (
                    _select_generation_step_updates(
                        x=x,
                        logits=logits,
                        mask_index=mask_index,
                        generated_index=generated_index,
                        prompt_lens=prompt_lens,
                        max_new_tokens=max_new_tokens,
                        total_len=total_len,
                        block_idx=block_index,
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

                if candidate_ids is not None:
                    candidate_ids.append(step_candidate_ids.clone())
                    transfer_indices.append(transfer_index.clone())
                    mask_indices.append(mask_index.clone())
                    remask_indices.append(remask_index.clone())
                    if top1_token_ids is not None and top1_confidences is not None:
                        top1_token_ids.append(step_top1_token_ids.clone())
                        top1_confidences.append(step_top1_confidences.clone())

                previous_state = x.clone()
                if remask_counts is not None:
                    remask_counts += remask_index.to(remask_counts.dtype)
                changed = _apply_generation_step_updates(
                    x=x,
                    candidate_ids=step_candidate_ids,
                    transfer_index=transfer_index,
                    remask_index=remask_index,
                    mask_id=mask_id,
                )
                if not changed:
                    break

                temporal_history = _append_generation_history(
                    temporal_history,
                    previous_state,
                    history_max_steps=history_max_steps,
                )
                if histories is not None:
                    histories.append(x.clone())
                global_step_idx += 1
            else:
                continue
            break

        if not return_dict:
            return x

        return LLaDATraceSamplerOutput(
            sequences=x,
            histories=histories,
            candidate_ids=candidate_ids,
            transfer_indices=transfer_indices,
            mask_indices=mask_indices,
            remask_indices=remask_indices,
            top1_token_ids=top1_token_ids,
            top1_confidences=top1_confidences,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-case inference for the LLaDA revision-history model."
    )
    p.add_argument("--pretrained", required=True, help="Path to the trained checkpoint.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="Prompt text (e.g. the math problem).")
    g.add_argument("--prompt-file", help="Path to a file containing the prompt text.")
    p.add_argument("--system-prompt", default=None)
    p.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Wrap the prompt with the tokenizer chat template.",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--allow-mask-token-candidate", default="True")
    p.add_argument("--reveal-token-num-per-step", type=int, default=1)
    p.add_argument("--remask-step-interval", type=int, default=1)
    p.add_argument("--max-remask-times", type=int, default=6)
    p.add_argument("--temporal-history-max-steps", type=int, default=6)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--output", default=None, help="Optional JSON file to write the result to.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    prompt = (
        args.prompt
        if args.prompt is not None
        else Path(args.prompt_file).read_text()
    )

    model_args = LLaDAHistoryTraceModelArguments(
        model_name_or_path=args.pretrained,
        temporal_history_max_steps=args.temporal_history_max_steps,
    )
    setattr(model_args, "dtype", args.dtype)

    model = load_llada_history_model(model_args)
    model.eval()
    tokenizer = load_llada_history_tokenizer(model_args)

    if args.apply_chat_template:
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": prompt})
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    else:
        input_ids = tokenizer(prompt)["input_ids"]

    config = LLaDAHistoryTraceEvalSamplerConfig(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_size=args.block_size,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        allow_mask_token_candidate=_coerce_bool(args.allow_mask_token_candidate),
        reveal_token_num_per_step=args.reveal_token_num_per_step,
        remask_step_interval=args.remask_step_interval,
        max_remask_times=args.max_remask_times,
        temporal_history_max_steps=args.temporal_history_max_steps,
        return_dict=False,
    )

    sampler = LLaDAHistoryTraceSampler(model=model, tokenizer=tokenizer)
    output = sampler.sample([input_ids], config=config)
    sequences = output if isinstance(output, torch.Tensor) else output.sequences

    prompt_len = len(input_ids)
    generated = sequences[0, prompt_len:].tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)

    print("=" * 80)
    print(text)
    print("=" * 80)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"prompt": prompt, "completion": text}, ensure_ascii=False, indent=2)
        )
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
