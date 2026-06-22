from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from peft import prepare_model_for_kbit_training
from transformers.models.auto import AutoConfig, AutoModel, AutoModelForMaskedLM

import dllm
from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM

HISTORY_CONFIG_FIELDS = (
    "temporal_history_decay",
    "temporal_history_max_steps",
    "temporal_distance_encoding",
    "base_model_name_or_path",
)


def _resolve_torch_dtype(
    dtype: str | torch.dtype | None,
) -> str | torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).strip()
    if hasattr(torch, normalized):
        return getattr(torch, normalized)
    return dtype


def _default_device_map():
    if (
        transformers.modeling_utils.is_deepspeed_zero3_enabled()
        or not torch.cuda.is_available()
    ):
        return None
    return {"": accelerate.PartialState().local_process_index}


def _default_quantization_config(load_in_4bit: bool, dtype) -> object | None:
    if not load_in_4bit or not transformers.utils.is_bitsandbytes_available():
        return None
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


class LLaDAHistoryConfig(LLaDAConfig):
    model_type = "llada_history"

    def __init__(
        self,
        temporal_history_decay: float = 0.0,
        temporal_history_max_steps: int | None = None,
        temporal_distance_encoding: str = "rope",
        base_model_name_or_path: str | None = None,
        **kwargs,
    ):
        kwargs["architectures"] = kwargs.get(
            "architectures",
            ["LLaDASyntheticRevisionHistoryModel"],
        )
        super().__init__(**kwargs)
        if temporal_history_decay < 0.0:
            raise ValueError("temporal_history_decay must be >= 0.")
        if (
            temporal_history_max_steps is not None
            and temporal_history_max_steps < 1
        ):
            raise ValueError("temporal_history_max_steps must be >= 1 when set.")
        if temporal_distance_encoding != "rope":
            raise ValueError("temporal_distance_encoding must be 'rope'.")

        self.temporal_history_decay = float(temporal_history_decay)
        self.temporal_history_max_steps = temporal_history_max_steps
        self.temporal_distance_encoding = temporal_distance_encoding
        self.base_model_name_or_path = base_model_name_or_path

    @classmethod
    def from_base_config(
        cls,
        base_config: LLaDAConfig,
        **extra_kwargs,
    ) -> "LLaDAHistoryConfig":
        base_payload = base_config.to_dict()
        base_payload.pop("model_type", None)
        base_payload.pop("architectures", None)
        return cls(**base_payload, **extra_kwargs)

    def history_config_dict(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in HISTORY_CONFIG_FIELDS
        }


class LLaDASyntheticRevisionHistoryModel(LLaDAModelLM):
    config_class = LLaDAHistoryConfig
    modules_to_save = {"history_output_norm"}

    def __init__(
        self,
        config: LLaDAHistoryConfig,
        model=None,
        init_params: bool = False,
    ):
        super().__init__(config=config, model=model, init_params=init_params)
        self.history_output_norm = nn.LayerNorm(config.d_model)
        self._init_weights(self.history_output_norm)

    @property
    def temporal_history_decay(self) -> float:
        return float(self.config.temporal_history_decay)

    @property
    def temporal_history_max_steps(self) -> int | None:
        return self.config.temporal_history_max_steps

    def _apply_rotary_distance_embedding(
        self,
        x: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        if distances.numel() == 0:
            return x

        hidden_size = x.shape[-1]
        half_dim = hidden_size // 2
        if half_dim == 0:
            return x

        freq_indices = torch.arange(half_dim, device=x.device, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (freq_indices / max(half_dim, 1)))
        angles = distances.to(torch.float32).unsqueeze(-1) * inv_freq
        cos_angles = torch.cos(angles).unsqueeze(-2)
        sin_angles = torch.sin(angles).unsqueeze(-2)

        x1 = x[..., :half_dim]
        x2 = x[..., half_dim : 2 * half_dim]
        rotated = torch.cat(
            [
                x1 * cos_angles - x2 * sin_angles,
                x1 * sin_angles + x2 * cos_angles,
            ],
            dim=-1,
        )
        if hidden_size % 2 == 1:
            rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
        return rotated

    def _history_residual(
        self,
        *,
        history_input_ids: torch.Tensor | None = None,
        history_distances: torch.Tensor | None = None,
        history_valid_mask: torch.Tensor | None = None,
        history_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if history_input_ids is None or history_input_ids.ndim != 3:
            return None
        if history_input_ids.shape[1] == 0:
            return None
        if history_distances is None or history_token_mask is None:
            return None

        history_embed = self.get_input_embeddings()(history_input_ids)
        history_embed = self._apply_rotary_distance_embedding(
            history_embed,
            history_distances,
        )

        embed_dtype = history_embed.dtype
        token_mask = history_token_mask.to(device=history_embed.device, dtype=embed_dtype)
        history_embed = history_embed * token_mask.unsqueeze(1).unsqueeze(-1)

        if history_valid_mask is not None:
            valid_mask = history_valid_mask.to(
                device=history_embed.device,
                dtype=embed_dtype,
            )
            history_embed = history_embed * valid_mask.unsqueeze(-1).unsqueeze(-1)

        if self.temporal_history_decay > 0.0:
            decay_weights = torch.exp(
                -self.temporal_history_decay
                * torch.clamp(history_distances.to(torch.float32) - 1.0, min=0.0)
            ).to(device=history_embed.device, dtype=embed_dtype)
            history_embed = history_embed * decay_weights.unsqueeze(-1).unsqueeze(-1)

        residual_sum = history_embed.sum(dim=1)
        norm_weight = self.history_output_norm.weight
        norm_bias = self.history_output_norm.bias
        norm_dtype = (
            norm_weight.dtype if norm_weight is not None else residual_sum.dtype
        )
        residual_sum = F.layer_norm(
            residual_sum.to(norm_dtype),
            self.history_output_norm.normalized_shape,
            weight=norm_weight,
            bias=norm_bias,
            eps=self.history_output_norm.eps,
        ).to(embed_dtype)
        return residual_sum

    def build_inputs_embeds(
        self,
        *,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        history_input_ids: torch.Tensor | None = None,
        history_distances: torch.Tensor | None = None,
        history_valid_mask: torch.Tensor | None = None,
        history_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            current_embeds = self.get_input_embeddings()(input_ids)
        else:
            current_embeds = inputs_embeds

        history_residual = self._history_residual(
            history_input_ids=history_input_ids,
            history_distances=history_distances,
            history_valid_mask=history_valid_mask,
            history_token_mask=history_token_mask,
        )
        if history_residual is None:
            return current_embeds
        return current_embeds + history_residual.to(current_embeds.dtype)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        past_key_values=None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position=None,
        history_input_ids: torch.Tensor | None = None,
        history_distances: torch.Tensor | None = None,
        history_valid_mask: torch.Tensor | None = None,
        history_token_mask: torch.Tensor | None = None,
    ):
        history_aware_embeds = self.build_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            history_input_ids=history_input_ids,
            history_distances=history_distances,
            history_valid_mask=history_valid_mask,
            history_token_mask=history_token_mask,
        )
        return super().forward(
            input_ids=None,
            inputs_embeds=history_aware_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    def save_pretrained(self, save_directory, *args, **kwargs):
        result = super().save_pretrained(save_directory, *args, **kwargs)
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        config_path = save_path / "config.json"
        if config_path.exists():
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            saved_config.pop("auto_map", None)
            config_path.write_text(
                json.dumps(saved_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        base_payload = self.config.to_dict().copy()
        for field_name in HISTORY_CONFIG_FIELDS:
            base_payload.pop(field_name, None)
        base_payload["model_type"] = "llada"
        base_payload["architectures"] = ["LLaDAModelLM"]

        (save_path / "base_model_config.json").write_text(
            json.dumps(base_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (save_path / "history_wrapper_config.json").write_text(
            json.dumps(self.config.history_config_dict(), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return result

    @classmethod
    def from_base_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        temporal_history_decay: float = 0.0,
        temporal_history_max_steps: int | None = None,
        temporal_distance_encoding: str = "rope",
        torch_dtype: str | torch.dtype | None = None,
        load_in_4bit: bool = False,
        attn_implementation: str | None = None,
    ) -> "LLaDASyntheticRevisionHistoryModel":
        base_config = LLaDAConfig.from_pretrained(pretrained_model_name_or_path)
        history_config = LLaDAHistoryConfig.from_base_config(
            base_config,
            temporal_history_decay=temporal_history_decay,
            temporal_history_max_steps=temporal_history_max_steps,
            temporal_distance_encoding=temporal_distance_encoding,
            base_model_name_or_path=str(pretrained_model_name_or_path),
        )
        dtype = _resolve_torch_dtype(torch_dtype)
        quant_config = _default_quantization_config(load_in_4bit, dtype)
        model = cls.from_pretrained(
            pretrained_model_name_or_path,
            config=history_config,
            torch_dtype=dtype,
            device_map=_default_device_map(),
            quantization_config=quant_config,
            attn_implementation=attn_implementation,
        )
        if load_in_4bit and quant_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=False,
            )
        return model


def load_llada_history_model(model_args) -> LLaDASyntheticRevisionHistoryModel:
    model_name_or_path = getattr(model_args, "model_name_or_path")
    raw_config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
    )
    dtype = _resolve_torch_dtype(getattr(model_args, "dtype", None))
    quant_config = _default_quantization_config(
        getattr(model_args, "load_in_4bit", False),
        dtype,
    )

    if getattr(raw_config, "model_type", None) == LLaDAHistoryConfig.model_type:
        model = LLaDASyntheticRevisionHistoryModel.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=_default_device_map(),
            quantization_config=quant_config,
            attn_implementation=getattr(model_args, "attn_implementation", None),
        )
        if getattr(model_args, "load_in_4bit", False) and quant_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=False,
            )
    else:
        model = LLaDASyntheticRevisionHistoryModel.from_base_pretrained(
            model_name_or_path,
            temporal_history_decay=float(
                getattr(model_args, "temporal_history_decay", 0.0)
            ),
            temporal_history_max_steps=getattr(
                model_args,
                "temporal_history_max_steps",
                None,
            ),
            temporal_distance_encoding=str(
                getattr(model_args, "temporal_distance_encoding", "rope")
            ),
            torch_dtype=dtype,
            load_in_4bit=bool(getattr(model_args, "load_in_4bit", False)),
            attn_implementation=getattr(model_args, "attn_implementation", None),
        )

    if getattr(model_args, "lora", False):
        modules_to_save = getattr(model_args, "modules_to_save", None)
        required_modules = ["history_output_norm"]
        if modules_to_save:
            existing = [value.strip() for value in modules_to_save.split(",") if value.strip()]
        else:
            existing = []
        for module_name in required_modules:
            if module_name not in existing:
                existing.append(module_name)
        model_args.modules_to_save = ",".join(existing)
        model = dllm.utils.load_peft(model, model_args)

    return model


def resolve_llada_history_base_model_name_or_path(model_name_or_path: str) -> str:
    checkpoint_path = Path(model_name_or_path)
    history_config_path = checkpoint_path / "history_wrapper_config.json"
    if history_config_path.exists():
        history_config = json.loads(history_config_path.read_text(encoding="utf-8"))
        base_model_name_or_path = history_config.get("base_model_name_or_path", None)
        if base_model_name_or_path:
            return dllm.utils.resolve_with_base_env(
                str(base_model_name_or_path),
                "BASE_MODELS_DIR",
            )

    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        base_model_name_or_path = config_payload.get("base_model_name_or_path", None)
        if base_model_name_or_path:
            return dllm.utils.resolve_with_base_env(
                str(base_model_name_or_path),
                "BASE_MODELS_DIR",
            )

    return dllm.utils.resolve_with_base_env(
        str(model_name_or_path),
        "BASE_MODELS_DIR",
    )


def load_llada_history_tokenizer(model_args):
    base_model_name_or_path = resolve_llada_history_base_model_name_or_path(
        getattr(model_args, "model_name_or_path")
    )
    return dllm.utils.get_tokenizer(
        model_args=SimpleNamespace(model_name_or_path=base_model_name_or_path)
    )


AutoConfig.register(LLaDAHistoryConfig.model_type, LLaDAHistoryConfig)
AutoModel.register(LLaDAHistoryConfig, LLaDASyntheticRevisionHistoryModel)
AutoModelForMaskedLM.register(
    LLaDAHistoryConfig,
    LLaDASyntheticRevisionHistoryModel,
)
