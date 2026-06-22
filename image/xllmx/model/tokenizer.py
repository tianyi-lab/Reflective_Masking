import logging
import os
from pathlib import Path
from typing import List, Optional

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer

__all__ = ["Tokenizer", "probe_tokenizer_path_from_pretrained"]


logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self, model_path: str):
        if model_path.endswith(".model"):
            self.tokenizer_type = "spm"
            assert os.path.isfile(model_path), model_path
            self.tokenizer = SentencePieceProcessor(model_file=model_path)
            logger.info(f"Reloaded SentencePiece model from {model_path}")

            self.bos_id: int = self.tokenizer.bos_id()
            self.eos_id: int = self.tokenizer.eos_id()
            assert self.tokenizer.vocab_size() == self.tokenizer.get_piece_size()
        else:
            self.tokenizer_type = "transformers"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"load HF transformers tokenizer from {model_path}")
            self.bos_id: int = self.tokenizer.bos_token_id
            if self.bos_id is None:
                self.bos_id = self.tokenizer.eos_token_id
            self.eos_id: int = self.tokenizer.eos_token_id
            assert self.eos_id is not None

        self._probe_tokenizer_style()

        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        if self.tokenizer_type == "transformers":
            t = self.tokenizer.encode(s, truncation=False, add_special_tokens=False)
        else:
            t = self.tokenizer.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def encode_segment(self, s: str):
        s = s.lstrip(" ")
        if self.need_space_before_segment:
            return self.encode(" " + s, bos=False, eos=False)
        else:
            return self.encode(s, bos=False, eos=False)

    def encode_wo_prefix_space(self, s: str):
        if self.need_space_before_segment:
            return self.encode(s, bos=False, eos=False)
        else:
            l_prefix = ["@", "\n", "\\", "=", ">", "`"]
            for prefix in l_prefix:
                prefix_tokens = self.encode(prefix, bos=False, eos=False)
                cat_tokens = self.encode(prefix + s, bos=False, eos=False)
                if cat_tokens[: len(prefix_tokens)] == prefix_tokens:
                    return cat_tokens[len(prefix_tokens) :]

            raise NotImplementedError(
                f"All prefixes are merged into {s} during tokenization,"
                f"This is wierd behavior, please open an issue to report this problem",
            )

    def _probe_tokenizer_style(self):
        sentence1 = self.encode("Hi my darling", bos=False, eos=False)
        sentence2 = self.encode("my darling", bos=False, eos=False)
        if sentence1[-len(sentence2) :] == sentence2:
            self.need_space_before_segment = False
        else:
            sentence3 = self.encode(" my darling", bos=False, eos=False)
            assert sentence1[-len(sentence3) :] == sentence3
            self.need_space_before_segment = True

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

    def save(self, save_dir: str):
        if self.tokenizer_type == "transformers":
            self.tokenizer.save_pretrained(save_dir)
        else:
            with open(Path(save_dir) / "tokenizer.model", "wb") as f:
                f.write(self.tokenizer.serialized_model_proto())

    @property
    def n_words(self):
        if self.tokenizer_type == "spm":
            return self.tokenizer.vocab_size()
        elif self.tokenizer_type == "transformers":
            return len(self.tokenizer)
        else:
            raise RuntimeError


def probe_tokenizer_path_from_pretrained(pretrained_path: str):
    tokenizer_path = None

    logger.info(f"trying to find sentencepiece-style tokenizer at {Path(pretrained_path) / 'tokenizer.model'}")
    if (Path(pretrained_path) / "tokenizer.model").exists():
        logger.info(f"Found {Path(pretrained_path) / 'tokenizer.model'}, use it.")
        tokenizer_path = str(Path(pretrained_path) / "tokenizer.model")
    else:
        logger.info("Not Found")

    if tokenizer_path is None:
        logger.info(
            f"trying to find huggingface-style tokenizer at "
            f"{Path(pretrained_path) / '(tokenizer.json, tokenizer_config.json)'}"
        )
        if (Path(pretrained_path) / "tokenizer.json").exists() and (
            Path(pretrained_path) / "tokenizer_config.json"
        ).exists():
            logger.info(f"Found {Path(pretrained_path) / '(tokenizer.json, tokenizer_config.json)'}, use them.")
            tokenizer_path = pretrained_path
        else:
            logger.info("Not Found")
    if tokenizer_path is None:
        logger.info("No usable tokenizer found")
    return tokenizer_path
