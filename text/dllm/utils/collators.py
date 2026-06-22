from dataclasses import dataclass
from typing import Any

import torch
import transformers


@dataclass
class CollatorWrapper:

    collator: Any

    def before(self, features):
        return features

    def after(self, outputs):
        return outputs

    def __call__(self, features, return_tensors=None):
        features = self.before(features)

        outputs = self.collator(features, return_tensors=return_tensors)

        outputs = self.after(outputs)
        return outputs

    def __getattr__(self, name: str):
        collator = self.__dict__.get("collator", None)
        if collator is not None:
            try:
                return getattr(collator, name)
            except AttributeError:
                pass

        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


@dataclass
class NoAttentionMaskWrapper(CollatorWrapper):

    def after(self, outputs):
        outputs.pop("attention_mask", None)
        return outputs


@dataclass
class PrependBOSWrapper(CollatorWrapper):

    bos_token_id: int | None = None
    label_pad_token_id: int = -100

    def after(self, outputs):
        assert self.bos_token_id
        input_ids = outputs.get("input_ids")

        bsz, _ = input_ids.shape

        bos = torch.full(
            (bsz, 1),
            self.bos_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([bos, input_ids], dim=1)
        outputs["input_ids"] = input_ids

        labels = outputs.get("labels", None)
        if labels is not None:
            ignore_labels = torch.full(
                (bsz, 1),
                self.label_pad_token_id,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([ignore_labels, labels], dim=1)
            outputs["labels"] = labels

        attention_mask = outputs.get("attention_mask", None)
        if attention_mask is not None:
            bos_attention = torch.ones(
                (bsz, 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([bos_attention, attention_mask], dim=1)
            outputs["attention_mask"] = attention_mask

        return outputs


@dataclass
class RandomTruncateWrapper(CollatorWrapper):

    random_length_ratio: float = 0.01
    label_pad_token_id: int = -100

    def after(self, outputs):
        if torch.rand(1) < self.random_length_ratio:
            input_ids = outputs["input_ids"]
            bsz, seq_len = input_ids.shape

            random_length = torch.randint(
                1, seq_len + 1, (1,), device=input_ids.device
            ).item()

            if "attention_mask" in outputs:
                outputs["attention_mask"][:, random_length:] = 0
            else:
                attention_mask = torch.ones(
                    (bsz, seq_len),
                    dtype=torch.long,
                    device=input_ids.device,
                )
                attention_mask[:, random_length:] = 0
                outputs["attention_mask"] = attention_mask

            if "labels" in outputs:
                outputs["labels"][:, random_length:] = self.label_pad_token_id

        return outputs


@dataclass
class RevisionDataCollator:

    tokenizer: transformers.PreTrainedTokenizer
    return_tensors: str = "pt"
    padding: bool = True
    pad_to_multiple_of: int | None = None

    def __call__(self, features, return_tensors=None):
        del return_tensors
        if not features:
            raise ValueError("RevisionDataCollator received an empty feature list.")

        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder

        batch_size = len(features)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("RevisionDataCollator requires tokenizer.pad_token_id.")

        input_ids = torch.full(
            (batch_size, max_length),
            pad_token_id,
            dtype=torch.long,
        )
        false_token_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        assistant_token_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        prompt_len = torch.zeros(batch_size, dtype=torch.long)

        for row_index, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row_index, :length] = torch.tensor(
                feature["input_ids"],
                dtype=torch.long,
            )
            false_token_mask[row_index, :length] = torch.tensor(
                feature["false_token_mask"],
                dtype=torch.bool,
            )
            assistant_token_mask[row_index, :length] = torch.tensor(
                feature["assistant_token_mask"],
                dtype=torch.bool,
            )
            prompt_len[row_index] = int(feature["prompt_len"])

        return {
            "input_ids": input_ids,
            "false_token_mask": false_token_mask,
            "assistant_token_mask": assistant_token_mask,
            "prompt_len": prompt_len,
        }


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")

    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        return_tensors="pt",
        padding=True,
    )

    collator = NoAttentionMaskWrapper(collator)

    samples = [
        {"input_ids": tokenizer("hello world")["input_ids"]},
        {"input_ids": tokenizer("goodbye")["input_ids"]},
    ]

    batch = collator(samples, return_tensors="pt")

    print("Batch keys:", batch.keys())
    print("input_ids:\n", batch["input_ids"])
    print("labels:\n", batch["labels"])

    assert "attention_mask" not in batch
    print("\nTest passed: attention_mask was removed.")
