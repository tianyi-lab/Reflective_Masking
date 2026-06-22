from importlib import import_module

__all__ = [
    "LLaDAConfig",
    "LLaDAMoEConfig",
    "LLaDAModelLM",
    "LLaDAMoEModelLM",
]

_LAZY_IMPORTS = {
    "LLaDAConfig": (".models.configuration_llada", "LLaDAConfig"),
    "LLaDAMoEConfig": (".models.configuration_lladamoe", "LLaDAMoEConfig"),
    "LLaDAModelLM": (".models.modeling_llada", "LLaDAModelLM"),
    "LLaDAMoEModelLM": (".models.modeling_lladamoe", "LLaDAMoEModelLM"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_path, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
