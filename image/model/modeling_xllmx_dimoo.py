import functools
import logging
import math
from typing import List
import torch.nn.functional as F
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig
from .modeling_llada import LLaDAModelLM
from .configuration_llada import LLaDAConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
__all__ = ["LLaDAForMultiModalGeneration"]

def create_attention_mask(original_lengths, max_tokens, device):
    batch_size = len(original_lengths)
    attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=device)
    for i, length in enumerate(original_lengths):
        attention_mask[i, :length] = 1
    return attention_mask

class LLaDAForMultiModalGeneration(LLaDAModelLM):
    config_class = LLaDAConfig
    base_model_prefix = "model"
    all_tied_weights_keys = {}

    def __init__(self, config: LLaDAConfig, *args, **kwargs):
        print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)
        self.all_tied_weights_keys = dict(getattr(self, "all_tied_weights_keys", {}) or {})

    def tie_weights(self, missing_keys=None, recompute_mapping: bool = True):
        del missing_keys, recompute_mapping
        return super().tie_weights()
    
    def forward(
        self,
        input_ids=None,
        labels=None,
        infer=False,
        input_embeddings=None,
        use_cache=False,
        to_compute_mask=None,
        cat='',
        **kwargs,
    ):
        if input_embeddings is not None:
            batch_size, seq_len = input_embeddings.shape[:2]
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            return LLaDAModelLM.forward(
                self,
                inputs_embeds=input_embeddings,
                attention_bias=attention_bias,
                use_cache=use_cache,
                to_compute_mask=to_compute_mask,
                cat=cat,
                **kwargs,
            )

        if infer:
            input_ids = input_ids.tolist()
        max_tokens = max([len(_) for _ in input_ids])
        original_lengths = [len(example) for example in input_ids]
        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device) 
        attention_mask = create_attention_mask(original_lengths, max_tokens, self.device)
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        output = LLaDAModelLM.forward(
            self,
            input_ids=input_ids,
            attention_bias=attention_bias,
            use_cache=use_cache,
            to_compute_mask=to_compute_mask,
            cat=cat,
            **kwargs,
        )
        if infer:
            return output
        
        labels = [label + [-100] * (max_tokens - len(label)) for label in labels]
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)
        logits = output.logits
        loss = F.cross_entropy(logits.contiguous().view(-1, logits.shape[-1]), labels.contiguous().view(-1), ignore_index=-100,)
        return loss
    
    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.transformer.blocks), self.model.transformer.ff_out]
        return modules
