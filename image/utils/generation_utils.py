import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(0.5 * math.pi * t)


def gumbel_noise(t: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if generator is None:
        u = torch.rand_like(t)
    else:
        u = torch.rand(t.shape, device=t.device, dtype=t.dtype, generator=generator)
    return -torch.log(-torch.log(u + 1e-20) + 1e-20)


def gumbel_max_sample(logits: torch.Tensor, tau: float = 1.0, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if tau == 0.0:
        return logits.argmax(dim=-1)
    g = gumbel_noise(logits, generator=generator)
    return (logits / tau + g).argmax(dim=-1)


def mask_by_random_topk(
    mask_len: torch.Tensor,
    probs: torch.Tensor,
    *,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.BoolTensor:
    g = gumbel_noise(probs, generator=generator)
    confidence = torch.log(probs.clamp_min(1e-20)) + temperature * g
    sorted_conf = torch.sort(confidence, dim=-1).values
    k = mask_len.long().unsqueeze(1).clamp_(0, probs.size(1) - 1)
    cut_off = torch.gather(sorted_conf, 1, k)
    return confidence < cut_off


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def setup_seed(seed: int):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
