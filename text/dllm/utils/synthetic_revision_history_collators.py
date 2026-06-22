from __future__ import annotations

from dataclasses import dataclass

import torch
import transformers


@dataclass
class SyntheticRevisionHistoryDataCollator:

    tokenizer: transformers.PreTrainedTokenizer
    return_tensors: str = "pt"
    padding: bool = True
    pad_to_multiple_of: int | None = None

    def __call__(self, features, return_tensors=None):
        del return_tensors
        if not features:
            raise ValueError(
                "SyntheticRevisionHistoryDataCollator received an empty feature list."
            )

        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder
        max_history_len = max(len(feature["history_input_ids"]) for feature in features)

        batch_size = len(features)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError(
                "SyntheticRevisionHistoryDataCollator requires tokenizer.pad_token_id."
            )

        input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
        current_input_ids = torch.full(
            (batch_size, max_length),
            pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        assistant_token_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        history_token_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        prompt_len = torch.zeros(batch_size, dtype=torch.long)
        current_progress = torch.zeros(batch_size, dtype=torch.long)

        history_input_ids = torch.full(
            (batch_size, max_history_len, max_length),
            pad_token_id,
            dtype=torch.long,
        )
        history_distances = torch.zeros((batch_size, max_history_len), dtype=torch.long)
        history_valid_mask = torch.zeros((batch_size, max_history_len), dtype=torch.bool)

        for row_index, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row_index, :length] = torch.tensor(
                feature["input_ids"],
                dtype=torch.long,
            )
            current_input_ids[row_index, :length] = torch.tensor(
                feature["current_input_ids"],
                dtype=torch.long,
            )
            attention_mask[row_index, :length] = 1
            assistant_token_mask[row_index, :length] = torch.tensor(
                feature["assistant_token_mask"],
                dtype=torch.bool,
            )
            history_token_mask[row_index, :length] = torch.tensor(
                feature["history_token_mask"],
                dtype=torch.bool,
            )
            prompt_len[row_index] = int(feature["prompt_len"])
            current_progress[row_index] = int(feature["current_progress"])

            feature_history = feature["history_input_ids"]
            feature_distances = feature["history_distances"]
            feature_valid = feature["history_valid_mask"]
            for history_index, history_ids in enumerate(feature_history):
                history_length = len(history_ids)
                history_input_ids[row_index, history_index, :history_length] = torch.tensor(
                    history_ids,
                    dtype=torch.long,
                )
            if feature_distances:
                history_distances[row_index, : len(feature_distances)] = torch.tensor(
                    feature_distances,
                    dtype=torch.long,
                )
            if feature_valid:
                history_valid_mask[row_index, : len(feature_valid)] = torch.tensor(
                    feature_valid,
                    dtype=torch.bool,
                )

        return {
            "input_ids": input_ids,
            "current_input_ids": current_input_ids,
            "attention_mask": attention_mask,
            "assistant_token_mask": assistant_token_mask,
            "history_input_ids": history_input_ids,
            "history_distances": history_distances,
            "history_valid_mask": history_valid_mask,
            "history_token_mask": history_token_mask,
            "prompt_len": prompt_len,
            "current_progress": current_progress,
        }
