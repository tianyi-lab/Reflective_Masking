
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import transformers

from dllm.utils.configs import TrainingArguments


def build_synthetic_revision_training_tensors(
    input_ids: torch.Tensor,
    corrupted_input_ids: torch.Tensor,
    assistant_token_mask: torch.Tensor,
    false_token_mask: torch.Tensor,
    masked_token_mask: torch.Tensor,
    *,
    mask_token_id: int,
    keep_loss_weight: float = 0.0,
) -> dict[str, torch.Tensor]:

    transition_mask = false_token_mask.bool()
    reveal_mask = masked_token_mask.bool() & assistant_token_mask.bool() & (~transition_mask)
    keep_mask = assistant_token_mask.bool() & (~transition_mask) & (~reveal_mask)

    targets = input_ids.clone()
    targets[transition_mask] = mask_token_id

    loss_weights = torch.zeros_like(input_ids, dtype=torch.float32)
    loss_weights[transition_mask] = 1.0
    loss_weights[reveal_mask] = 1.0
    if keep_loss_weight > 0.0:
        loss_weights[keep_mask] = keep_loss_weight

    supervision_mask = loss_weights > 0
    edit_mask = transition_mask | reveal_mask
    return {
        "noised_input_ids": corrupted_input_ids,
        "targets": targets,
        "transition_mask": transition_mask,
        "reveal_mask": reveal_mask,
        "keep_mask": keep_mask,
        "edit_mask": edit_mask,
        "loss_weights": loss_weights,
        "supervision_mask": supervision_mask,
    }


def masked_accuracy_stats(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mask.bool()
    correct = (predictions == targets) & mask
    return correct.sum(dtype=torch.float32), mask.sum(dtype=torch.float32)


def masked_weighted_loss_stats(
    token_nll: torch.Tensor,
    loss_weights: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mask.bool()
    masked_weights = loss_weights * mask.to(loss_weights.dtype)
    weight_sum = masked_weights.sum(dtype=torch.float32)
    weighted_loss_sum = (token_nll * masked_weights).sum(dtype=torch.float32)
    return weighted_loss_sum / weight_sum.clamp_min(1.0), weight_sum


class SyntheticRevisionMetricsCallback(transformers.TrainerCallback):
    def __init__(
        self,
        trainer: "SyntheticRevisionMDLMTrainer",
        metric_names: tuple[str, ...],
        splits: tuple[str, ...] = ("train", "eval"),
    ):
        super().__init__()
        self.trainer = trainer
        self.accelerator = trainer.accelerator
        self.splits = splits
        self.metric_names = metric_names
        device = self.accelerator.device
        self.metrics: dict[str, dict[str, torchmetrics.aggregation.MeanMetric]] = {}
        for split in splits:
            split_metrics: dict[str, torchmetrics.aggregation.MeanMetric] = {}
            for name in metric_names:
                metric = torchmetrics.aggregation.MeanMetric(sync_on_compute=True)
                metric.to(device)
                metric.reset()
                split_metrics[name] = metric
            self.metrics[split] = split_metrics

    @staticmethod
    def key_for(split: str, name: str) -> str:
        return name if split == "train" else f"{split}_{name}"

    @torch.no_grad()
    def update(
        self,
        split: str,
        name: str,
        value: torch.Tensor | float,
        weight: torch.Tensor | float = 1.0,
    ) -> None:
        self.metrics[split][name].update(value, weight=weight)

    @torch.no_grad()
    def finalize(self, split: str) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, metric in self.metrics[split].items():
            computed = metric.compute()
            metric.reset()
            if isinstance(computed, torch.Tensor) and torch.isnan(computed):
                computed = torch.tensor(0.0, device=computed.device)
            values[name] = float(computed.item())
        return values

    def on_log(self, args, state, control, logs=None, **kwargs):
        return control


@dataclass
class SyntheticRevisionMDLMConfig(TrainingArguments):
    loss_norm_type: str = "token"
    right_shift_logits: bool = False
    keep_loss_weight: float = 0.0


class SyntheticRevisionMDLMTrainer(transformers.Trainer):
    def __init__(
        self,
        args: SyntheticRevisionMDLMConfig,
        *pargs,
        **kwargs,
    ):
        processing_class = kwargs.pop("processing_class", None)
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is None and processing_class is not None:
            kwargs["tokenizer"] = processing_class
        super().__init__(args=args, *pargs, **kwargs)

        self.loss_norm_type = args.loss_norm_type
        self.right_shift_logits = args.right_shift_logits
        self.keep_loss_weight = args.keep_loss_weight
        if processing_class is not None:
            self.processing_class = processing_class
        elif getattr(self, "processing_class", None) is None and getattr(self, "tokenizer", None) is not None:
            self.processing_class = self.tokenizer

        self.meter = SyntheticRevisionMetricsCallback(
            trainer=self,
            metric_names=(
                "loss",
                "mask_loss",
                "reveal_loss",
                "keep_loss",
                "transition_accuracy",
                "reveal_accuracy",
                "keep_accuracy",
                "edit_accuracy",
            ),
        )
        self.add_callback(self.meter)

    def _get_processor(self):
        processor = getattr(self, "processing_class", None)
        if processor is None:
            processor = getattr(self, "tokenizer", None)
        if processor is None:
            raise ValueError(
                "SyntheticRevisionMDLMTrainer requires a tokenizer or processing_class."
            )
        return processor

    def _preprocess_inputs(self, inputs):
        if not self.right_shift_logits:
            return inputs

        bos_token_id = getattr(self._get_processor(), "bos_token_id", None)
        if bos_token_id is None:
            raise ValueError("right_shift_logits=True requires a tokenizer bos_token_id.")

        batch_size = inputs["input_ids"].shape[0]
        bos = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=inputs["input_ids"].dtype,
            device=inputs["input_ids"].device,
        )
        for key in ("input_ids", "corrupted_input_ids"):
            inputs[key] = torch.cat([bos, inputs[key]], dim=1)

        zero_bool = torch.zeros(
            (batch_size, 1),
            dtype=torch.bool,
            device=inputs["input_ids"].device,
        )
        for key in ("false_token_mask", "assistant_token_mask", "masked_token_mask"):
            if key in inputs:
                inputs[key] = torch.cat([zero_bool, inputs[key]], dim=1)

        if "attention_mask" in inputs:
            bos_attention = torch.ones(
                (batch_size, 1),
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            )
            inputs["attention_mask"] = torch.cat(
                [bos_attention, inputs["attention_mask"]],
                dim=1,
            )

        if "prompt_len" in inputs:
            inputs["prompt_len"] = inputs["prompt_len"] + 1
        return inputs

    def _postprocess_outputs(self, outputs):
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        split = (
            "eval"
            if any(str(key).startswith("eval_") for key in logs.keys())
            else "train"
        )
        metric_values = self.meter.finalize(split)
        for name, value in metric_values.items():
            logs[self.meter.key_for(split, name)] = value
        super().log(logs, start_time=start_time)

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()
        return (loss.detach(), logits, None)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        del kwargs
        processor = self._get_processor()
        assert processor.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)

        input_ids = inputs["input_ids"]
        corrupted_input_ids = inputs["corrupted_input_ids"]
        false_token_mask = inputs["false_token_mask"].bool()
        assistant_token_mask = inputs["assistant_token_mask"].bool()
        masked_token_mask = inputs["masked_token_mask"].bool()
        attention_mask = inputs.get("attention_mask", None)

        if corrupted_input_ids.shape != input_ids.shape:
            raise ValueError("corrupted_input_ids must have the same shape as input_ids.")
        if torch.any(false_token_mask & (~assistant_token_mask)):
            raise ValueError("false_token_mask must be a subset of assistant_token_mask.")
        if torch.any(masked_token_mask & (~assistant_token_mask)):
            raise ValueError("masked_token_mask must be a subset of assistant_token_mask.")
        if torch.any(false_token_mask & masked_token_mask):
            raise ValueError("false_token_mask and masked_token_mask must be disjoint.")
        if torch.any(corrupted_input_ids[masked_token_mask] != processor.mask_token_id):
            raise ValueError(
                "masked_token_mask positions in corrupted_input_ids must equal mask_token_id."
            )

        batch_size = input_ids.shape[0]
        tensors = build_synthetic_revision_training_tensors(
            input_ids=input_ids,
            corrupted_input_ids=corrupted_input_ids,
            assistant_token_mask=assistant_token_mask,
            false_token_mask=false_token_mask,
            masked_token_mask=masked_token_mask,
            mask_token_id=processor.mask_token_id,
            keep_loss_weight=self.keep_loss_weight,
        )

        outputs = model(
            input_ids=tensors["noised_input_ids"],
            attention_mask=attention_mask,
        )
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        token_nll = F.cross_entropy(
            logits.transpose(1, 2),
            tensors["targets"],
            reduction="none",
        )
        weighted_token_nll = token_nll * tensors["loss_weights"]
        mask_loss, mask_weight = masked_weighted_loss_stats(
            token_nll,
            tensors["loss_weights"],
            tensors["transition_mask"],
        )
        reveal_loss, reveal_weight = masked_weighted_loss_stats(
            token_nll,
            tensors["loss_weights"],
            tensors["reveal_mask"],
        )
        keep_loss, keep_weight = masked_weighted_loss_stats(
            token_nll,
            tensors["loss_weights"],
            tensors["keep_mask"],
        )

        if self.loss_norm_type == "token":
            normalizer = tensors["loss_weights"].sum().clamp_min(1.0)
            loss = weighted_token_nll.sum() / normalizer
        elif self.loss_norm_type == "sequence":
            normalizer = tensors["loss_weights"].sum(-1).clamp_min(1.0)
            loss = (weighted_token_nll.sum(-1) / normalizer).mean()
        elif self.loss_norm_type == "batch":
            loss = weighted_token_nll.sum() / batch_size
        else:
            raise ValueError("Invalid loss_norm_type.")

        predictions = logits.argmax(dim=-1)
        transition_correct, transition_total = masked_accuracy_stats(
            predictions,
            tensors["targets"],
            tensors["transition_mask"],
        )
        reveal_correct, reveal_total = masked_accuracy_stats(
            predictions,
            tensors["targets"],
            tensors["reveal_mask"],
        )
        keep_correct, keep_total = masked_accuracy_stats(
            predictions,
            tensors["targets"],
            tensors["keep_mask"],
        )
        edit_correct, edit_total = masked_accuracy_stats(
            predictions,
            tensors["targets"],
            tensors["edit_mask"],
        )

        split = "train" if model.training else "eval"
        self.meter.update(split, "loss", loss.detach(), weight=1.0)
        self.meter.update(split, "mask_loss", mask_loss.detach(), weight=mask_weight)
        self.meter.update(
            split,
            "reveal_loss",
            reveal_loss.detach(),
            weight=reveal_weight,
        )
        self.meter.update(split, "keep_loss", keep_loss.detach(), weight=keep_weight)
        self.meter.update(
            split,
            "transition_accuracy",
            transition_correct / transition_total.clamp_min(1.0),
            weight=transition_total,
        )
        self.meter.update(
            split,
            "reveal_accuracy",
            reveal_correct / reveal_total.clamp_min(1.0),
            weight=reveal_total,
        )
        self.meter.update(
            split,
            "keep_accuracy",
            keep_correct / keep_total.clamp_min(1.0),
            weight=keep_total,
        )
        self.meter.update(
            split,
            "edit_accuracy",
            edit_correct / edit_total.clamp_min(1.0),
            weight=edit_total,
        )

        return (loss, outputs) if return_outputs else loss


def build_synthetic_revision_history_training_tensors(
    input_ids: torch.Tensor,
    current_input_ids: torch.Tensor,
    assistant_token_mask: torch.Tensor,
    *,
    mask_token_id: int,
    keep_loss_weight: float = 0.0,
) -> dict[str, torch.Tensor]:

    assistant_token_mask = assistant_token_mask.bool()
    transition_mask = (
        assistant_token_mask
        & (current_input_ids != input_ids)
        & (current_input_ids != mask_token_id)
    )
    reveal_mask = assistant_token_mask & (current_input_ids == mask_token_id)
    keep_mask = assistant_token_mask & (~transition_mask) & (~reveal_mask)

    targets = input_ids.clone()
    targets[transition_mask] = mask_token_id

    loss_weights = torch.zeros_like(input_ids, dtype=torch.float32)
    loss_weights[transition_mask] = 1.0
    loss_weights[reveal_mask] = 1.0
    if keep_loss_weight > 0.0:
        loss_weights[keep_mask] = keep_loss_weight

    supervision_mask = loss_weights > 0
    edit_mask = transition_mask | reveal_mask
    return {
        "model_input_ids": current_input_ids,
        "targets": targets,
        "transition_mask": transition_mask,
        "reveal_mask": reveal_mask,
        "keep_mask": keep_mask,
        "edit_mask": edit_mask,
        "loss_weights": loss_weights,
        "supervision_mask": supervision_mask,
    }


@dataclass
class SyntheticRevisionHistoryMDLMConfig(SyntheticRevisionMDLMConfig):
    pass


class SyntheticRevisionHistoryMDLMTrainer(SyntheticRevisionMDLMTrainer):
    def _preprocess_inputs(self, inputs):
        if not self.right_shift_logits:
            return inputs

        bos_token_id = getattr(self._get_processor(), "bos_token_id", None)
        if bos_token_id is None:
            raise ValueError("right_shift_logits=True requires a tokenizer bos_token_id.")

        batch_size = inputs["input_ids"].shape[0]
        bos = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=inputs["input_ids"].dtype,
            device=inputs["input_ids"].device,
        )
        for key in ("input_ids", "current_input_ids"):
            inputs[key] = torch.cat([bos, inputs[key]], dim=1)

        zero_bool = torch.zeros(
            (batch_size, 1),
            dtype=torch.bool,
            device=inputs["input_ids"].device,
        )
        for key in ("assistant_token_mask", "history_token_mask"):
            if key in inputs:
                inputs[key] = torch.cat([zero_bool, inputs[key]], dim=1)

        if "history_input_ids" in inputs:
            history_len = inputs["history_input_ids"].shape[1]
            bos_history = torch.full(
                (batch_size, history_len, 1),
                bos_token_id,
                dtype=inputs["history_input_ids"].dtype,
                device=inputs["history_input_ids"].device,
            )
            inputs["history_input_ids"] = torch.cat(
                [bos_history, inputs["history_input_ids"]],
                dim=-1,
            )

        if "attention_mask" in inputs:
            bos_attention = torch.ones(
                (batch_size, 1),
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            )
            inputs["attention_mask"] = torch.cat(
                [bos_attention, inputs["attention_mask"]],
                dim=1,
            )

        if "prompt_len" in inputs:
            inputs["prompt_len"] = inputs["prompt_len"] + 1
        return inputs

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        del kwargs
        processor = self._get_processor()
        assert processor.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)

        input_ids = inputs["input_ids"]
        current_input_ids = inputs["current_input_ids"]
        assistant_token_mask = inputs["assistant_token_mask"].bool()
        attention_mask = inputs.get("attention_mask", None)
        history_input_ids = inputs.get("history_input_ids", None)
        history_distances = inputs.get("history_distances", None)
        history_valid_mask = inputs.get("history_valid_mask", None)
        history_token_mask = inputs.get("history_token_mask", None)

        if current_input_ids.shape != input_ids.shape:
            raise ValueError("current_input_ids must have the same shape as input_ids.")
        if torch.any((~assistant_token_mask) & (current_input_ids == processor.mask_token_id)):
            raise ValueError(
                "Only assistant positions may remain masked in current_input_ids."
            )
        if history_input_ids is not None and history_token_mask is None:
            raise ValueError("history_token_mask is required when history_input_ids is provided.")

        batch_size = input_ids.shape[0]
        tensors = build_synthetic_revision_history_training_tensors(
            input_ids=input_ids,
            current_input_ids=current_input_ids,
            assistant_token_mask=assistant_token_mask,
            mask_token_id=processor.mask_token_id,
            keep_loss_weight=self.keep_loss_weight,
        )

        outputs = model(
            input_ids=tensors["model_input_ids"],
            attention_mask=attention_mask,
            history_input_ids=history_input_ids,
            history_distances=history_distances,
            history_valid_mask=history_valid_mask,
            history_token_mask=history_token_mask,
        )
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        token_nll = F.cross_entropy(
            logits.transpose(1, 2),
            tensors["targets"],
            reduction="none",
        )
        weighted_token_nll = token_nll * tensors["loss_weights"]
        mask_loss, mask_weight = masked_weighted_loss_stats(
            token_nll,
            tensors["loss_weights"],
            tensors["transition_mask"],
        )
        reveal_loss, reveal_weight = masked_weighted_loss_stats(
            token_nll,
            tensors["loss_weights"],
            tensors["reveal_mask"],
        )
        keep_loss, keep_weight = masked_weighted_loss_stats(
            token_nll,
            tensors["loss_weights"],
            tensors["keep_mask"],
        )

        if self.loss_norm_type == "token":
            normalizer = tensors["loss_weights"].sum().clamp_min(1.0)
            loss = weighted_token_nll.sum() / normalizer
        elif self.loss_norm_type == "sequence":
            normalizer = tensors["loss_weights"].sum(-1).clamp_min(1.0)
            loss = (weighted_token_nll.sum(-1) / normalizer).mean()
        elif self.loss_norm_type == "batch":
            loss = weighted_token_nll.sum() / batch_size
        else:
            raise ValueError("Invalid loss_norm_type.")

        predictions = logits.argmax(dim=-1)
        transition_correct, transition_total = masked_accuracy_stats(
            predictions,
            tensors["targets"],
            tensors["transition_mask"],
        )
        reveal_correct, reveal_total = masked_accuracy_stats(
            predictions,
            tensors["targets"],
            tensors["reveal_mask"],
        )
        keep_correct, keep_total = masked_accuracy_stats(
            predictions,
            tensors["targets"],
            tensors["keep_mask"],
        )
        edit_correct, edit_total = masked_accuracy_stats(
            predictions,
            tensors["targets"],
            tensors["edit_mask"],
        )

        split = "train" if model.training else "eval"
        self.meter.update(split, "loss", loss.detach(), weight=1.0)
        self.meter.update(split, "mask_loss", mask_loss.detach(), weight=mask_weight)
        self.meter.update(
            split,
            "reveal_loss",
            reveal_loss.detach(),
            weight=reveal_weight,
        )
        self.meter.update(split, "keep_loss", keep_loss.detach(), weight=keep_weight)
        self.meter.update(
            split,
            "transition_accuracy",
            transition_correct / transition_total.clamp_min(1.0),
            weight=transition_total,
        )
        self.meter.update(
            split,
            "reveal_accuracy",
            reveal_correct / reveal_total.clamp_min(1.0),
            weight=reveal_total,
        )
        self.meter.update(
            split,
            "keep_accuracy",
            keep_correct / keep_total.clamp_min(1.0),
            weight=keep_total,
        )
        self.meter.update(
            split,
            "edit_accuracy",
            edit_correct / edit_total.clamp_min(1.0),
            weight=edit_total,
        )

        return (loss, outputs) if return_outputs else loss
