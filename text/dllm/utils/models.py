from types import SimpleNamespace

import accelerate
import torch
import transformers
from peft import prepare_model_for_kbit_training

from dllm.utils.configs import ModelArguments, TrainingArguments
from dllm.utils.utils import disable_caching_allocator_warmup, load_peft, print_main


def _resolve_model_config_and_trust_remote_code(
    model_name_or_path: str,
) -> tuple[transformers.PretrainedConfig | None, bool]:
    local_model_types = {
        "llada",
        "lladamoe",
        "llada2_moe",
        "dream",
        "Dream",
        "dream_history",
    }

    config_dict, _ = transformers.PretrainedConfig.get_config_dict(model_name_or_path)
    model_type = config_dict.get("model_type")
    if model_type in local_model_types:
        if model_type in {"llada", "lladamoe"}:
            import dllm.pipelines.llada.models
        elif model_type == "llada2_moe":
            import dllm.pipelines.llada2.models
        elif model_type in {"dream", "Dream", "dream_history"}:
            import dllm.pipelines.dream.models

        local_config = transformers.AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=False,
        )
        return local_config, False

    remote_config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model_type = getattr(remote_config, "model_type", None)
    if model_type not in local_model_types:
        return remote_config, True

    if model_type in {"llada", "lladamoe"}:
        import dllm.pipelines.llada.models
    elif model_type == "llada2_moe":
        import dllm.pipelines.llada2.models
    elif model_type in {"dream", "Dream", "dream_history"}:
        import dllm.pipelines.dream.models

    local_config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
    )
    return local_config, False


def _get_local_model_class(model_type: str | None):
    if model_type == "llada":
        from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM

        return LLaDAModelLM
    if model_type == "lladamoe":
        from dllm.pipelines.llada.models.modeling_lladamoe import LLaDAMoEModelLM

        return LLaDAMoEModelLM
    if model_type == "llada2_moe":
        from dllm.pipelines.llada2.models.modeling_llada2_moe import LLaDA2MoeModelLM

        return LLaDA2MoeModelLM
    if model_type in {"dream", "Dream"}:
        from dllm.pipelines.dream.models.modeling_dream import DreamModel

        return DreamModel
    if model_type == "dream_history":
        from dllm.pipelines.dream.models.history_wrapper import (
            DreamSyntheticRevisionHistoryModel,
        )

        return DreamSyntheticRevisionHistoryModel
    return None


def get_model(
    model_args: ModelArguments | None = None,
    config: transformers.PretrainedConfig | None = None,
    **kwargs,
) -> transformers.PreTrainedModel:
    model_args = model_args or ModelArguments()
    model_name_or_path = kwargs.get(
        "model_name_or_path", getattr(model_args, "model_name_or_path", None)
    )
    dtype = kwargs.get("dtype", getattr(model_args, "dtype", "bfloat16"))
    load_in_4bit = kwargs.get(
        "load_in_4bit", getattr(model_args, "load_in_4bit", False)
    )
    attn_implementation = kwargs.get(
        "attn_implementation", getattr(model_args, "attn_implementation", None)
    )

    if config is None:
        config, trust_remote_code = _resolve_model_config_and_trust_remote_code(
            model_name_or_path
        )
    else:
        trust_remote_code = True

    device_map = (
        {"": accelerate.PartialState().local_process_index}
        if not transformers.modeling_utils.is_deepspeed_zero3_enabled()
        and torch.cuda.is_available()
        else None
    )

    quant_config = None
    if load_in_4bit and transformers.utils.is_bitsandbytes_available():
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    params = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "quantization_config": quant_config,
        "attn_implementation": attn_implementation,
        "config": config,
        "trust_remote_code": trust_remote_code,
    }

    model_type = getattr(config, "model_type", None)
    local_model_cls = None if trust_remote_code else _get_local_model_class(model_type)

    if local_model_cls is not None:
        local_params = dict(params)
        local_params.pop("trust_remote_code", None)
        model = local_model_cls.from_pretrained(model_name_or_path, **local_params)
    else:
        try:
            model = transformers.AutoModelForMaskedLM.from_pretrained(
                model_name_or_path, **params
            )
        except Exception:
            model = transformers.AutoModel.from_pretrained(model_name_or_path, **params)

    no_split_modules = getattr(model, "_no_split_modules", None)
    if no_split_modules:
        available_module_classes = {type(module).__name__ for module in model.modules()}
        normalized_no_split_modules = []
        for module_name in no_split_modules:
            if (
                module_name in available_module_classes
                and module_name not in normalized_no_split_modules
            ):
                normalized_no_split_modules.append(module_name)
        if normalized_no_split_modules != list(no_split_modules):
            print_main(
                "Normalizing model._no_split_modules from "
                f"{list(no_split_modules)} to {normalized_no_split_modules}"
            )
            model._no_split_modules = normalized_no_split_modules

    if load_in_4bit and quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    model = load_peft(model, model_args)

    return model


def get_tokenizer(
    model_args: ModelArguments | None = None, **kwargs
) -> transformers.PreTrainedTokenizer:
    from transformers import BertPreTrainedModel, RobertaPreTrainedModel

    try:
        from transformers import ModernBertPreTrainedModel
    except ImportError:
        ModernBertPreTrainedModel = None

    try:
        from dllm.pipelines.a2d import (
            A2DLlamaLMHeadModel,
            A2DQwen2LMHeadModel,
            A2DQwen3LMHeadModel,
        )
    except Exception:
        A2DLlamaLMHeadModel = None
        A2DQwen2LMHeadModel = None
        A2DQwen3LMHeadModel = None

    try:
        from dllm.pipelines.dream.models.modeling_dream import DreamModel
    except Exception:
        DreamModel = None
    try:
        from dllm.pipelines.dream.models.history_wrapper import (
            DreamSyntheticRevisionHistoryModel,
        )
    except Exception:
        DreamSyntheticRevisionHistoryModel = None

    try:
        from dllm.pipelines.llada2.models.modeling_llada2_moe import LLaDA2MoeModelLM
    except Exception:
        LLaDA2MoeModelLM = None

    try:
        from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    except Exception:
        LLaDAModelLM = None

    try:
        from dllm.pipelines.llada.models.modeling_lladamoe import LLaDAMoEModelLM
    except Exception:
        LLaDAMoEModelLM = None

    model_args = model_args or ModelArguments()
    model_name_or_path = kwargs.get(
        "model_name_or_path", getattr(model_args, "model_name_or_path", None)
    )

    model_cfg, config_trust_remote_code = _resolve_model_config_and_trust_remote_code(
        model_name_or_path
    )
    model_type = getattr(model_cfg, "model_type", None)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        trust_remote_code=config_trust_remote_code,
    )

    assert tokenizer.eos_token is not None or tokenizer.pad_token is not None

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.eos_token:
        tokenizer.eos_token = tokenizer.pad_token
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.pad_token

    try:
        model_cls = transformers.AutoModel._model_mapping[type(model_cfg)]
    except KeyError:
        if model_type == "llada":
            model_cls = LLaDAModelLM
        elif model_type == "lladamoe":
            model_cls = LLaDAMoEModelLM
        elif model_type == "llada2_moe":
            model_cls = LLaDA2MoeModelLM
        else:
            model_cls = None

    if model_cls is None:
        model_cls = type(model_cfg)

    if model_type == "llada" or (LLaDAModelLM is not None and issubclass(model_cls, LLaDAModelLM)):
        tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
        tokenizer.eot_token = "<|eot_id|>"
        tokenizer.chat_template = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor %}
{% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
    elif model_type in {"lladamoe", "llada2_moe"} or (
        any(cls is not None for cls in (LLaDAMoEModelLM, LLaDA2MoeModelLM))
        and issubclass(
            model_cls,
            tuple(
                cls
                for cls in (LLaDAMoEModelLM, LLaDA2MoeModelLM)
                if cls is not None
            ),
        )
    ):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|role_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif (
        model_type in {"dream", "Dream", "dream_history"}
        or (DreamModel is not None and issubclass(model_cls, DreamModel))
        or (
            DreamSyntheticRevisionHistoryModel is not None
            and issubclass(model_cls, DreamSyntheticRevisionHistoryModel)
        )
    ):
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif issubclass(
        model_cls,
        tuple(
            cls
            for cls in (
                BertPreTrainedModel,
                RobertaPreTrainedModel,
                ModernBertPreTrainedModel,
            )
            if cls is not None
        ),
    ):
        tokenizer.eot_token = "[/Answer]"
        tokenizer.chat_template = """\
{% if messages[0]['role'] == 'system' %}
[SYS]
{{ messages[0]['content'] | trim }}
[/SYS]

{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif -%}
{%- for message in loop_messages %}
{% if message['role'] == 'user' %}
[Question]
{{ message['content'] | trim }}
[/Question]

{% elif message['role'] == 'assistant' %}
[Answer]
{{ message['content'] | trim }}
[/Answer]

{% endif %}
{% endfor -%}
{%- if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
[Answer]
{% endif %}
"""
    elif A2DLlamaLMHeadModel is not None and issubclass(model_cls, A2DLlamaLMHeadModel):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|eot_id|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif (
        A2DQwen2LMHeadModel is not None
        and A2DQwen3LMHeadModel is not None
        and issubclass(model_cls, (A2DQwen2LMHeadModel, A2DQwen3LMHeadModel))
    ):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
        _orig_apply_chat_template = tokenizer.apply_chat_template

        def _apply_chat_template(*args, **kwargs):
            if "enable_thinking" not in kwargs:
                kwargs["enable_thinking"] = False
            try:
                return _orig_apply_chat_template(*args, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return _orig_apply_chat_template(*args, **kwargs)

        tokenizer.apply_chat_template = _apply_chat_template
    else:
        print_main("no tokenizer customization for model config:", type(model_cfg))
    return tokenizer
