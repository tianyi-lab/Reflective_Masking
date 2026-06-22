import logging
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dllm.utils.configs import ModelArguments, DataArguments, TrainingArguments

import pprint

import accelerate
import peft
import torch
import transformers


def resolve_with_base_env(path: str, env_name: str) -> str:
    base = os.getenv(env_name, "").strip()
    if not base:
        return path
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path

    candidate = os.path.join(base.rstrip("/"), path.lstrip("/"))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Path not found: {candidate}")


@contextmanager
def init_device_context_manager(device: str | torch.device | None = None):
    if transformers.integrations.is_deepspeed_zero3_enabled():
        yield
        return

    if device is None:
        try:
            from accelerate import PartialState

            idx = PartialState().local_process_index
        except Exception:
            idx = 0
        device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, int):
        device = f"cuda:{device}"

    try:
        torch.set_default_device(device)
        yield
    finally:
        torch.set_default_device("cpu")


def print_main(*args, **kwargs):
    if accelerate.PartialState().is_main_process:
        print(*args, **kwargs)


def pprint_main(*args, **kwargs):
    if accelerate.PartialState().is_main_process:
        pprint.pprint(*args, **kwargs)


def load_peft(
    model: transformers.PreTrainedModel, model_args: "ModelArguments"
) -> transformers.PreTrainedModel:
    if not getattr(model_args, "lora", False):
        return model
    target_modules = (
        model_args.target_modules.split(",") if model_args.target_modules else None
    )
    if (
        target_modules
        and len(target_modules) == 1
        and target_modules[0].strip() == "all-linear"
    ):
        target_modules = target_modules[0]
    modules_to_save = (
        model_args.modules_to_save.split(",") if model_args.modules_to_save else None
    )
    peft_config = peft.LoraConfig(
        r=model_args.r,
        target_modules=target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias=model_args.bias,
        modules_to_save=modules_to_save,
    )
    model = peft.get_peft_model(model, peft_config)
    if accelerate.PartialState().is_main_process:
        print(model)
        model.print_trainable_parameters()
    return model


def print_args_main(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
):
    print_main("\n===== Parsed arguments =====")
    for name, args in [
        ("model_args", model_args),
        ("data_args", data_args),
        ("training_args", training_args),
    ]:
        d = asdict(args)
        short = {k: d[k] for k in list(d)}
        print_main(f"{name}:")
        pprint_main(short, width=100, compact=True, sort_dicts=False)
    print_main("============================\n")


def print_args(args):
    print_main("\n===== Parsed arguments =====")
    d = asdict(args)
    short = {k: d[k] for k in list(d)}
    pprint_main(short, width=100, compact=True, sort_dicts=False)
    print_main("============================\n")


def disable_caching_allocator_warmup():
    try:
        from transformers import modeling_utils as _mu

        def _noop(*args, **kwargs):
            return

        _mu.caching_allocator_warmup = _noop
    except Exception:
        pass


def disable_dataset_progress_bar_except_main():
    from datasets.utils.logging import disable_progress_bar, enable_progress_bar

    if accelerate.PartialState().is_main_process:
        enable_progress_bar()
    else:
        disable_progress_bar()


def initial_training_setup(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
):
    transformers.set_seed(training_args.seed)
    disable_caching_allocator_warmup()
    disable_dataset_progress_bar_except_main()
    if getattr(data_args, "disable_caching", False):
        disable_dataset_caching()


def disable_dataset_caching():
    from datasets import disable_caching

    disable_caching()
    tmp_root = f"/tmp/hf_cache_rank{accelerate.PartialState().process_index}"
    os.environ["HF_DATASETS_CACHE"] = tmp_root
    os.environ["HF_DATASETS_TEMP_DIR"] = tmp_root
    os.makedirs(tmp_root, exist_ok=True)


def parse_spec(spec: str):

    def _parse_kv_string(s: str) -> dict:
        return dict(part.split("=", 1) for part in s.split(",") if "=" in part)

    s = spec.strip()

    m = re.search(r"\[(.*?)\]$", s)
    bracket_kvs = {}
    numeric_kvs = {}
    if m:
        bracket = m.group(1).strip()
        if bracket:
            for part in bracket.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError(
                        f"Invalid entry '{part}' in '{spec}' (expected key:value)."
                    )
                key, value = part.split(":", 1)
                key = key.strip()
                value = value.strip()

                if re.fullmatch(r"\d(?:_?\d)*", value):
                    numeric_kvs[key] = int(value.replace("_", ""))
                else:
                    bracket_kvs[key] = value

        s = s[: m.start()].rstrip()

    name = None
    if "=" in s:
        kv_dict = dict(_parse_kv_string(s))
    else:
        kv_dict = {}
        if s:
            name = s

    kv_dict.update(bracket_kvs)
    kv_dict.update(numeric_kvs)

    return name, kv_dict


def get_default_logger(name):
    logger = logging.getLogger(name)
    if accelerate.PartialState().is_main_process:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt=(
                "\x1b[38;5;110m[%(asctime)s "
                "\x1b[38;5;174m%(levelname)s "
                "\x1b[38;5;109m%(name)s"
                "/%(lineno)d-%(processName)s\x1b[38;5;110m] "
                "\x1b[0m%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
