import random
import warnings
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING

import datasets
import torch
import transformers

if TYPE_CHECKING:
    from dllm.utils.configs import DataArguments, ModelArguments, TrainingArguments


def tokenize_and_group(
    examples,
    tokenizer,
    text_field: str = "text",
    seq_length: int = 1024,
    insert_eos: bool = False,
    drop_tail: bool = True,
    add_special_tokens: bool = False,
):
    tokenized = tokenizer(examples[text_field], add_special_tokens=add_special_tokens)
    ids = tokenized["input_ids"]

    if insert_eos:
        eos_id = getattr(tokenizer, "eos_token_id")
        assert eos_id
        ids = [seq + ([] if (seq and seq[-1] == eos_id) else [eos_id]) for seq in ids]

    concatenated = list(chain.from_iterable(ids))
    if not concatenated:
        return {"input_ids": [], "labels": []}

    if drop_tail:
        total_len = (len(concatenated) // seq_length) * seq_length
        concatenated = concatenated[:total_len]
    else:
        total_len = len(concatenated)

    chunks = [concatenated[i : i + seq_length] for i in range(0, total_len, seq_length)]

    return {
        "input_ids": chunks,
        "labels": [c[:] for c in chunks],
    }


def clip_row(row: dict, max_length: int, truncation: str = "right") -> dict:
    for key in ("input_ids", "labels", "attention_mask"):
        if key in row:
            if truncation == "right":
                row[key] = row[key][:max_length]
            elif truncation == "left":
                row[key] = row[key][-max_length:]
            else:
                raise NotImplementedError
    return row


def post_process_dataset(
    dataset: datasets.DatasetDict, data_args: "DataArguments"
) -> datasets.DatasetDict:
    if data_args.truncation == "filter":
        return dataset.filter(
            lambda row: len(row["input_ids"]) <= data_args.max_length,
            num_proc=data_args.num_proc,
            desc=f"Filtering samples with length <= {data_args.max_length}",
        )
    elif data_args.truncation == "right":
        if "prompt_len" in dataset.column_names["train"]:
            dataset = dataset.filter(
                lambda row: row["prompt_len"] <= data_args.max_length,
                num_proc=data_args.num_proc,
                desc=f"Filtering samples with `prompt_len` <= {data_args.max_length}",
            )
        return dataset.map(
            lambda row: clip_row(row, data_args.max_length, truncation="right"),
            num_proc=data_args.num_proc,
            desc=f"Right-truncating samples to max_length={data_args.max_length}",
        )
    else:
        raise NotImplementedError


def clip_row_streaming(row: dict, max_length: int, truncation: str = "right") -> dict:
    if truncation not in {"right", "left"}:
        raise NotImplementedError(f"Unknown truncation: {truncation}")

    def clip(seq):
        return seq[:max_length] if truncation == "right" else seq[-max_length:]

    def clip_preserve_prompt(seq, prompt_len: int):
        prompt = seq[:prompt_len]
        resp = seq[prompt_len:]
        budget = max(0, max_length - len(prompt))
        resp = resp[:budget] if truncation == "right" else resp[-budget:]
        return prompt + resp

    prompt_len = row.get("prompt_len", None)
    for k in ("input_ids", "labels", "attention_mask"):
        if k in row and isinstance(row[k], list):
            row[k] = (
                clip_preserve_prompt(row[k], prompt_len)
                if isinstance(prompt_len, int) and prompt_len >= 0
                else clip(row[k])
            )
    return row


def post_process_dataset_streaming(
    dataset: datasets.IterableDatasetDict,
    data_args: "DataArguments",
) -> datasets.IterableDatasetDict:

    def _train_has_prompt_len_streaming(dataset: datasets.IterableDatasetDict) -> bool:
        it = dataset["train"].take(1)
        try:
            ex = next(iter(it))
        except StopIteration:
            return False
        return "prompt_len" in ex

    mode = data_args.truncation
    max_len = data_args.max_length

    if mode == "filter":
        def keep_if_short(row):
            if (
                "input_ids" in row
                and isinstance(row["input_ids"], list)
                and len(row["input_ids"]) <= max_len
            ):
                yield row

        return datasets.IterableDatasetDict(
            {name: ds.map(keep_if_short) for name, ds in dataset.items()}
        )

    elif mode == "right":
        ds_out = dataset

        if _train_has_prompt_len_streaming(ds_out):

            def keep_if_prompt_fits(row):
                pl = row.get("prompt_len", None)
                if isinstance(pl, int) and pl <= max_len:
                    yield row
                elif pl is None:
                    return

            ds_out = datasets.IterableDatasetDict(
                {name: ds.map(keep_if_prompt_fits) for name, ds in ds_out.items()}
            )

        def clip_right(row):
            return clip_row(row, max_len, truncation="right")

        return datasets.IterableDatasetDict(
            {name: ds.map(clip_right) for name, ds in ds_out.items()}
        )

    else:
        raise NotImplementedError


def default_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    prompt_response_tokens = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False
    )
    labels = prompt_response_tokens.copy()

    if mask_prompt_loss:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            "prompt_len": len(prompt_tokens),
        }

    return {"input_ids": prompt_response_tokens, "labels": labels}


def prepend_bos(
    batch: dict,
    bos_token_id: int,
    label_pad_token_id: int = -100,
):
    assert bos_token_id is not None, "bos_token_id must be provided"

    input_ids = batch.get("input_ids")
    bsz, _ = input_ids.shape

    bos = torch.full(
        (bsz, 1),
        bos_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    batch["input_ids"] = torch.cat([bos, input_ids], dim=1)

    labels = batch.get("labels")
    if labels is not None:
        ignore_labels = torch.full(
            (bsz, 1),
            label_pad_token_id,
            dtype=labels.dtype,
            device=labels.device,
        )
        batch["labels"] = torch.cat([ignore_labels, labels], dim=1)

    attn = batch.get("attention_mask")
    if attn is not None:
        bos_attention = torch.ones(
            (bsz, 1),
            dtype=attn.dtype,
            device=attn.device,
        )
        batch["attention_mask"] = torch.cat([bos_attention, attn], dim=1)

    return batch
