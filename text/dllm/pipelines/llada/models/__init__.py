from .configuration_llada import LLaDAConfig
from .modeling_llada import LLaDAModelLM
from .configuration_lladamoe import LLaDAMoEConfig
from .modeling_lladamoe import LLaDAMoEModelLM

__all__ = [
    "LLaDAConfig",
    "LLaDAModelLM",
    "LLaDAMoEConfig",
    "LLaDAMoEModelLM",
]

try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada", LLaDAConfig)
    AutoModel.register(LLaDAConfig, LLaDAModelLM)
    AutoModelForMaskedLM.register(LLaDAConfig, LLaDAModelLM)

    AutoConfig.register("lladamoe", LLaDAMoEConfig)
    AutoModel.register(LLaDAMoEConfig, LLaDAMoEModelLM)
    AutoModelForMaskedLM.register(LLaDAMoEConfig, LLaDAMoEModelLM)

except ImportError:
    pass
