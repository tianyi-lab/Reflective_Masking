
from dataclasses import dataclass

import torch
import torch.distributions as dists
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.pipelines.dream.models.generation_utils import top_k_logits, top_p_logits


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class DreamSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 20
    max_length: int = (
        None
    )
    steps: int = 512
    eps: float = 1e-3
    alg: str = "origin"
    alg_temp: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    stochastic_transfer: bool = False
    right_shift_logits: bool = True
    cfg_scale: float = 0.0


@dataclass
class DreamSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor] | list[list[int]],
        config: DreamSamplerConfig | None = None,
        generation_tokens_hook_func=lambda step, x, logits: x,
        generation_logits_hook_func=lambda step, x, logits: logits,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        if config is None:
            config = DreamSamplerConfig()

        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        steps = kwargs.get("steps", config.steps)
        eps = kwargs.get("eps", config.eps)
        alg = kwargs.get("alg", config.alg)
        alg_temp = kwargs.get("alg_temp", config.alg_temp)
        temperature = kwargs.get("temperature", config.temperature)
        top_p = kwargs.get("top_p", config.top_p)
        top_k = kwargs.get("top_k", config.top_k)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)

        mask_token_id = self.tokenizer.mask_token_id
        eos_token_id = self.tokenizer.eos_token_id

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
        x = torch.full((B, T), eos_token_id, dtype=torch.long, device=self.model.device)

        seq_lens = []
        for i, p in enumerate(inputs):
            total_len = prompt_lens[i] + max_new_tokens
            seq_lens.append(total_len)
            start = T - total_len
            x[i, start : start + prompt_lens[i]] = p
            x[i, start + prompt_lens[i] : T] = mask_token_id

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for j, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[j, -L:] = 1

        if attention_mask is not None and torch.any(attention_mask == 0):
            pos_id = attention_mask.long().cumsum(-1) - 1
            pos_id.masked_fill_(attention_mask == 0, 1)
        else:
            pos_id = None

        mask_index = x == mask_token_id
        num_transfer_tokens_list = get_num_transfer_tokens(
            mask_index=mask_index,
            steps=steps,
            scheduler=self.scheduler,
            stochastic=stochastic_transfer,
        )
        effective_steps = num_transfer_tokens_list.size(1)

        prompt_index = attention_mask.bool() & (
            torch.arange(T, device=x.device).unsqueeze(0) < T - max_new_tokens
        )

        x = generation_tokens_hook_func(None, x, None)
        histories = [x.clone()] if return_dict else None
        for i in range(effective_steps):
            mask_index = x == mask_token_id

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_token_id
                x_ = torch.cat([x, un_x], dim=0)
                attention_mask_cfg = torch.cat([attention_mask, attention_mask], dim=0)
                pos_id_cfg = (
                    torch.cat([pos_id, pos_id], dim=0) if pos_id is not None else None
                )
                logits = self.model(x_, attention_mask_cfg, pos_id_cfg).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = self.model(x, attention_mask, pos_id).logits

            if right_shift_logits:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]

            if alg == "maskgit_plus":
                confidence, x0 = sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
                )
            elif alg == "topk_margin":
                confidence, x0 = sample_tokens(
                    mask_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    margin_confidence=True,
                )
            elif alg == "entropy":
                confidence, x0 = sample_tokens(
                    mask_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    neg_entropy=True,
                )
            else:
                raise RuntimeError(f"Unknown alg: {alg}")

            full_confidence = torch.full_like(
                x, -torch.inf, device=self.model.device, dtype=logits.dtype
            )
            full_confidence[mask_index] = confidence

            for j in range(full_confidence.shape[0]):
                number_transfer_tokens = num_transfer_tokens_list[j, i]
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(
                            full_confidence[j], number_transfer_tokens
                        )
                    else:
                        fc = full_confidence[j] / alg_temp
                        fc = F.softmax(fc, dim=-1)
                        transfer_index = torch.multinomial(
                            fc, num_samples=number_transfer_tokens
                        )

                    x_ = torch.full_like(x, mask_token_id, device=self.model.device)
                    x_[mask_index] = x0.clone()
                    x[j, transfer_index] = x_[j, transfer_index]

            x = generation_tokens_hook_func(i, x, logits)
            if histories is not None:
                histories.append(x.clone())

        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor] | list[list[int]],
        config,
        generation_tokens_hook_func=lambda step, x, logits: x,
        generation_logits_hook_func=lambda step, x, logits: logits,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        steps = kwargs.get("steps", config.steps)
        eps = kwargs.get("eps", config.eps)
        alg = kwargs.get("alg", config.alg)
        alg_temp = kwargs.get("alg_temp", config.alg_temp)
        temperature = kwargs.get("temperature", config.temperature)
        top_p = kwargs.get("top_p", config.top_p)
        top_k = kwargs.get("top_k", config.top_k)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)

        mask_token_id = self.tokenizer.mask_token_id
        eos_token_id = self.tokenizer.eos_token_id

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        x = torch.full((B, T), eos_token_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            L = seq_lens[i]
            x[i, -L:] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for j, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[j, -L:] = 1

        if attention_mask is not None and torch.any(attention_mask == 0):
            pos_id = attention_mask.long().cumsum(-1) - 1
            pos_id.masked_fill_(attention_mask == 0, 1)
        else:
            pos_id = None

        mask_index = x == mask_token_id
        num_transfer_tokens_list = get_num_transfer_tokens(
            mask_index=mask_index,
            steps=steps,
            scheduler=self.scheduler,
            stochastic=stochastic_transfer,
        )
        effective_steps = num_transfer_tokens_list.size(1)

        x = generation_tokens_hook_func(None, x, None)
        histories = [x.clone()] if return_dict else None
        for i in range(effective_steps):
            mask_index = x == mask_token_id

            logits = self.model(x, attention_mask, pos_id).logits
            if right_shift_logits:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]

            if alg == "maskgit_plus":
                confidence, x0 = sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
                )
            elif alg == "topk_margin":
                confidence, x0 = sample_tokens(
                    mask_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    margin_confidence=True,
                )
            elif alg == "entropy":
                confidence, x0 = sample_tokens(
                    mask_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    neg_entropy=True,
                )
            else:
                raise RuntimeError(f"Unknown alg: {alg}")

            full_confidence = torch.full_like(
                x, -torch.inf, device=self.model.device, dtype=logits.dtype
            )
            full_confidence[mask_index] = confidence

            for j in range(B):
                number_transfer_tokens = num_transfer_tokens_list[j, i]
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(
                            full_confidence[j], number_transfer_tokens
                        )
                    else:
                        fc = full_confidence[j] / alg_temp
                        fc = F.softmax(fc, dim=-1)
                        transfer_index = torch.multinomial(
                            fc, num_samples=number_transfer_tokens
                        )

                    x_ = torch.full_like(x, mask_token_id, device=self.model.device)
                    x_[mask_index] = x0.clone()
                    x[j, transfer_index] = x_[j, transfer_index]

            x = generation_tokens_hook_func(i, x, logits)
            if histories is not None:
                histories.append(x.clone())

        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)
