
import os
from dataclasses import dataclass

import accelerate
import transformers

import dllm
from dllm.core.trainers.synthetic_revision_history_mdlm import (
    SyntheticRevisionHistoryMDLMConfig,
    SyntheticRevisionHistoryMDLMTrainer,
)
from dllm.data.synthetic_revision_history import load_synthetic_revision_history_dataset
from dllm.pipelines.llada.models.history_wrapper import load_llada_history_model
from dllm.utils.synthetic_revision_history_collators import (
    SyntheticRevisionHistoryDataCollator,
)

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "CKPT/LLaDA-8B-Instruct"
    temporal_history_decay: float = 0.0
    temporal_history_max_steps: int | None = None
    temporal_distance_encoding: str = "rope"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = (
        "data/preprocessed/"
        "hendrycks_math/max_total_2048-max_new_2048_top_k_8_corruption_rate_0.3_candidate_sampling_uniform/"
        "synthetic-wrong-history_steps_6_history_6"
    )


@dataclass
class TrainingArguments(SyntheticRevisionHistoryMDLMConfig):
    output_dir: str = (
        "output/LLaDA-8B-Instruct/"
        "hendrycks_math-synthetic-revision-history"
    )
    group_by_length: bool = True
    num_train_epochs: float = 5
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    keep_loss_weight: float = 1.0


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    model = load_llada_history_model(model_args)
    if training_args.gradient_checkpointing:
        supports_gc = bool(
            getattr(model, "supports_gradient_checkpointing", False)
            or getattr(model, "_supports_gradient_checkpointing", False)
        )
        if not supports_gc:
            logger.warning(
                "Model %s does not support gradient checkpointing; disabling it.",
                model.__class__.__name__,
            )
            training_args.gradient_checkpointing = False
            training_args.gradient_checkpointing_kwargs = None
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    with accelerate.PartialState().local_main_process_first():
        dataset = load_synthetic_revision_history_dataset(data_args.dataset_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start synthetic revision history training...")
    trainer = SyntheticRevisionHistoryMDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=SyntheticRevisionHistoryDataCollator(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
