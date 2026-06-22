
from __future__ import annotations

from datasets import DatasetDict, load_from_disk

from dllm.utils.utils import resolve_with_base_env

SYNTHETIC_REVISION_HISTORY_REQUIRED_COLUMNS = (
    "input_ids",
    "current_input_ids",
    "prompt_len",
    "assistant_token_mask",
    "history_input_ids",
    "history_distances",
    "history_valid_mask",
    "history_token_mask",
    "current_progress",
)


def validate_synthetic_revision_history_dataset(dataset: DatasetDict) -> DatasetDict:
    for split, split_dataset in dataset.items():
        missing_columns = [
            column
            for column in SYNTHETIC_REVISION_HISTORY_REQUIRED_COLUMNS
            if column not in split_dataset.column_names
        ]
        if missing_columns:
            raise ValueError(
                f"Preprocessed synthetic revision history split '{split}' is missing columns: "
                f"{missing_columns}. Found columns: {split_dataset.column_names}"
            )
    return dataset


def load_synthetic_revision_history_dataset(dataset_args: str) -> DatasetDict:
    dataset_path = resolve_with_base_env(dataset_args, "BASE_DATASETS_DIR")
    dataset = load_from_disk(dataset_path)
    return validate_synthetic_revision_history_dataset(dataset)
