from typing import Any
from dataclasses import dataclass

import torch

from dllm.core.trainers import MDLMConfig, MDLMTrainer


def cart_weight(
    masked_mask: torch.Tensor, t: torch.Tensor, p: float = 0.3
) -> torch.Tensor:
    b, l = masked_mask.shape
    device = masked_mask.device

    idx = torch.arange(l, device=device)
    dist_matrix = idx[None, :] - idx[:, None]
    geo_matrix = (
        torch.log(torch.tensor(p, device=device))
        + (dist_matrix.abs() - 1).clamp(min=0) * torch.log(torch.tensor(1 - p, device=device))
    ).exp() * 0.5
    geo_matrix.masked_fill_(dist_matrix == 0, 0.0)

    valid_mask = (~masked_mask).float()
    weights = valid_mask @ geo_matrix.T
    weights = weights * masked_mask.float()
    return weights


class DreamTrainer(MDLMTrainer):

    @dataclass
    class DreamConfig(MDLMConfig):
        loss_weight_type: str = "cart[geo_p:0.3]"
        right_shift_logits: bool = True

        def __post_init__(self):
            super().__post_init__()
            assert self.right_shift_logits

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        masked_mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.loss_weight_type.startswith("cart"):
            import re

            match = re.search(r"geo_p:(0\.\d+)", self.loss_weight_type)
            geo_p = float(match.group(1)) if match else 0.3
            loss_weights = cart_weight(masked_mask, t, p=geo_p)
        else:
            loss_weights = super()._compute_loss_weights(
                t=t,
                inputs=inputs,
                masked_mask=masked_mask,
                *args,
                **kwargs,
            )
        return loss_weights
