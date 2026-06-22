from .configuration_dream import DreamConfig
from .modeling_dream import DreamModel
from .history_wrapper import DreamHistoryConfig, DreamSyntheticRevisionHistoryModel

__all__ = [
    "DreamConfig",
    "DreamModel",
    "DreamHistoryConfig",
    "DreamSyntheticRevisionHistoryModel",
]

try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    try:
        AutoConfig.register("Dream", DreamConfig)
    except ValueError:
        pass
    try:
        AutoModel.register(DreamConfig, DreamModel)
    except ValueError:
        pass
    try:
        AutoModelForMaskedLM.register(DreamConfig, DreamModel)
    except ValueError:
        pass
except ImportError:
    pass
