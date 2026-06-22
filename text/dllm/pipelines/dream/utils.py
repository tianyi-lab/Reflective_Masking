from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import transformers






@dataclass
class DreamSFTCollator(transformers.DataCollatorForSeq2Seq):

    perbatch_cutoff: bool = True
    resp_cutoff_ratio: float = 0.0

    def apply_perbatch_cutoff(self, features):
        resp_lens = torch.tensor(
            [len(f["input_ids"]) - f["prompt_len"] for f in features], dtype=torch.long
        )
        kept_len = int(np.random.choice(resp_lens))
        for f, r_len in zip(features, resp_lens):
            remove_len = max(r_len - kept_len, 0)
            if remove_len > 0:
                for key in ["input_ids", "labels", "attention_mask"]:
                    if key in f:
                        f[key] = f[key][:-remove_len]
        return features

    def apply_resp_cutoff(self, batch, features):
        orig_seq_lens = [len(f["input_ids"]) for f in features]
        resp_lens = torch.tensor(
            [len(f["input_ids"]) - f["prompt_len"] for f in features], dtype=torch.long
        )
        min_resp_len = resp_lens.min().item()
        if min_resp_len <= 1:
            return batch

        cutoff_len = int(np.random.randint(1, min_resp_len))
        new_seq_len = max(orig_seq_lens) - cutoff_len

        for key in ["input_ids", "labels", "attention_mask"]:
            if key in batch:
                batch[key] = batch[key][:, :new_seq_len].contiguous()
        return batch

    def __call__(self, features, return_tensors=None):
        if self.perbatch_cutoff:
            features = self.apply_perbatch_cutoff(features)

        base = [
            {k: f[k] for k in ("input_ids", "labels", "attention_mask") if k in f}
            for f in features
        ]
        batch = super().__call__(base, return_tensors=return_tensors)

        if (
            not self.perbatch_cutoff
            and self.resp_cutoff_ratio > 0
            and np.random.rand() < self.resp_cutoff_ratio
        ):
            batch = self.apply_resp_cutoff(batch, features)

        batch.pop("prompt_len", None)
        return batch
