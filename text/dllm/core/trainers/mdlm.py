
from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.utils.configs import TrainingArguments
from dllm.utils.data import prepend_bos
from .utils import NLLMetric, PPLMetric, OnEvaluateMetricsCallback


@dataclass
class MDLMConfig(TrainingArguments):
    time_epsilon: float = 1e-3
    loss_weight_type: str = "scheduler"
    loss_norm_type: str = "token"
    right_shift_logits: bool = False


class MDLMTrainer(transformers.Trainer):

    def __init__(
        self,
        args: MDLMConfig,
        scheduler: BaseAlphaScheduler | None = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)

        if not (0.0 < args.time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")

        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.time_epsilon = args.time_epsilon
        self.loss_weight_type = args.loss_weight_type
        self.loss_norm_type = args.loss_norm_type
        self.right_shift_logits = args.right_shift_logits

        self.meter = OnEvaluateMetricsCallback(
            trainer=self,
            splits=("train", "eval"),
            metrics={"nll": NLLMetric(), "ppl": PPLMetric()},
        )
        self.add_callback(self.meter)

    def _preprocess_inputs(self, inputs):
        if self.right_shift_logits:
            labels = inputs.get("labels", None)

            if labels is not None:
                if torch.all(labels[:, 0] == -100):
                    return inputs

            inputs = prepend_bos(
                inputs,
                bos_token_id=self.processing_class.bos_token_id,
                label_pad_token_id=-100,
            )
        return inputs

    def _postprocess_outputs(self, outputs):
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        b, l = inputs["input_ids"].shape
        if self.loss_weight_type == "scheduler":
            loss_weights = self.scheduler.weight(t).unsqueeze(1).repeat(1, l)
        elif self.loss_weight_type == "uniform":
            loss_weights = torch.ones_like(inputs["input_ids"])
        else:
            raise NotImplementedError
        return loss_weights

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()

        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        maskable_mask = labels != -100

        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)

        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"

        token_nll = F.cross_entropy(
            logits.transpose(1, 2),
            input_ids,
            reduction="none",
        )
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=maskable_mask.to(dtype=logits.dtype).detach(),
        )

        if self.loss_norm_type == "token":
            token_nll /= maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll /= maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            token_nll /= b
        else:
            raise ValueError("Invalid loss_norm_type.")
        loss = token_nll.sum()

        return (loss, outputs) if return_outputs else loss
