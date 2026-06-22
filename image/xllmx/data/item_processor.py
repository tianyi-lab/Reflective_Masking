from abc import ABC, abstractmethod
import copy
import logging
from typing import Any, Callable, Dict, List, Tuple, Union

from xllmx.model.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class LabelAllZeroError(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"LabelAllZeroError: {self.message}"


class ItemProcessorBase(ABC):
    @abstractmethod
    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        raise NotImplementedError

    def predict_item_token_length(self, data_item: dict) -> int:
        return 1


class MMConvItemProcessor(ItemProcessorBase):
    def __init__(
        self,
        transform: Dict[str, Callable[[Any], Dict]],
        media_symbols: List[str],
        tokenizer: str | Tokenizer,
        conv_template,
    ):
        self.transform = transform
        logger.info(f"transform:\n{self.transform}")

        self.media_symbols = media_symbols
        logger.info(f"media_symbols:\n{self.media_symbols}")

        if isinstance(tokenizer, str):
            self.tokenizer = Tokenizer(model_path=tokenizer)
        else:
            self.tokenizer = copy.deepcopy(tokenizer)

        self.tokenizer.tokenizer.add_tokens(media_symbols)
        self.d_media_symbol2token = {}
        self.d_media_token2symbol = {}
        for media_symbol in media_symbols:
            tokenized_symbol = self.tokenizer.encode(media_symbol, bos=False, eos=False)
            assert len(tokenized_symbol) == 1
            self.d_media_symbol2token[media_symbol] = tokenized_symbol[0]
            self.d_media_token2symbol[tokenized_symbol[0]] = media_symbol

        self.implicit_at_beginning = False
        self.conv_template = conv_template

    def collect_and_process_media(self, data_item):
        d_media = {}
        for media_symbol in self.media_symbols:
            if media_symbol in data_item:
                l_media = data_item[media_symbol]
            elif media_symbol.lstrip("<|").rstrip("|>") in data_item:
                l_media = data_item[media_symbol.lstrip("<|").rstrip("|>")]
            else:
                l_media = []
            if not isinstance(l_media, list):
                l_media = [l_media]

            d_media[media_symbol] = []
            for media in l_media:
                media = self.transform[media_symbol](media)
                assert isinstance(media, Dict)
                media["type"] = media_symbol
                d_media[media_symbol].append(media)

        return d_media

    def replace_media_token_with_media(
        self, tokens: List[int], labels: Union[List[int], None], d_media: Dict[str, List]
    ):
        d_media_counter = {key: 0 for key in d_media}
        for i, t in enumerate(tokens):
            if t in self.d_media_token2symbol:
                media_symbol = self.d_media_token2symbol[t]
                media = d_media[media_symbol][d_media_counter[media_symbol]]
                d_media_counter[media_symbol] += 1
                tokens[i] = media
                media["to_predict"] = labels[i] > 0

        assert all([d_media_counter[key] == len(d_media[key]) for key in d_media])

        if labels is not None:
            return tokens, labels
        else:
            return tokens

    @staticmethod
    def insert_implicit_media_symbol_in_q1(conv_list: List[Dict], d_media: Dict):
        conv_list = copy.deepcopy(conv_list)

        for media_symbol, l_media in d_media.items():
            media_symbol_count = "".join([_["value"] for _ in conv_list if _["value"] is not None]).count(media_symbol)
            if media_symbol_count > 0:
                assert media_symbol_count == len(
                    l_media
                ), f"{media_symbol_count} {media_symbol} exists in text, but {len(l_media)} actual media are given"
            else:
                conv_list[0]["value"] = (media_symbol + " ") * len(l_media) + conv_list[0]["value"]

        return conv_list

    @staticmethod
    def insert_implicit_media_symbol_at_beginning(conv: str, d_media: Dict):
        conv = copy.deepcopy(conv)

        for media_symbol, l_media in d_media.items():
            media_symbol_count = conv.count(media_symbol)
            if media_symbol_count > 0:
                assert media_symbol_count == len(
                    l_media
                ), f"{media_symbol_count} {media_symbol} exists in text, but {len(l_media)} actual media are given"
            else:
                conv = (media_symbol + " ") * len(l_media) + conv

        return conv

    def preprocess_item(self, data_item):
        return data_item

    def add_speaker_and_signal(self, source: List):
        conv = self.conv_template()

        for i, sentence in enumerate(source):
            from_str = sentence["from"]
            if i % 2 == 0:
                assert from_str.lower() in ["human"]
                role = conv.roles[0]
            elif i % 2 == 1:
                assert from_str.lower() in ["gpt", "assistant"]
                role = conv.roles[1]
            else:
                raise ValueError(f"unknown dialog role: {from_str.lower()}")

            value = sentence["value"]

            conv.append_message(role, value)

        processed = conv.process()
        conversation, pieces = processed["conv"], processed["pieces"]

        return conversation, pieces

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        data_item = self.preprocess_item(data_item)

        d_media = self.collect_and_process_media(data_item)

        source = data_item["conversations"]

        if not self.implicit_at_beginning:
            source = self.insert_implicit_media_symbol_in_q1(source, d_media)

        conversation, pieces = self.add_speaker_and_signal(source)

        if self.implicit_at_beginning:
            conversation = self.insert_implicit_media_symbol_at_beginning(conversation, d_media)

        tokens = self.tokenizer.encode(conversation, bos=True, eos=False)
        labels = [-100 for _ in tokens]

        for media_symbol, l_media in d_media.items():
            media_token = self.d_media_symbol2token[media_symbol]
            media_token_count = tokens.count(media_token)
            assert media_token_count == len(l_media), (
                f"{media_token_count} {media_token} (for {media_symbol}) exists in tokenized conversation, "
                f"but {len(l_media)} actual media are given"
            )

        check_pos = 0
        for i, p in enumerate(pieces):
            if i == 0:
                tokenized_value = self.tokenizer.encode(p["data"], bos=True, eos=False)
            else:
                tokenized_value = self.tokenizer.encode_wo_prefix_space(p["data"])

            assert (
                tokens[check_pos : check_pos + len(tokenized_value)] == tokenized_value
            ), "inconsistent complete conversation and corresponding piece after tokenization"

            if p["predict"]:
                labels[check_pos : check_pos + len(tokenized_value)] = tokenized_value

            check_pos = check_pos + len(tokenized_value)

        if training_mode and all([_ <= 0 for _ in labels]):
            raise LabelAllZeroError()

        tokens, labels = self.replace_media_token_with_media(tokens, labels, d_media)

        assert len(tokens) == len(labels)

        if training_mode:
            return tokens, labels
        else:
            return tokens

    def predict_item_token_length(self, data_item: dict) -> int:

        if "conversations" in data_item:
            print("Ok")
            return sum([len(_["value"]) for _ in data_item["conversations"]])
        else:
            return 1
