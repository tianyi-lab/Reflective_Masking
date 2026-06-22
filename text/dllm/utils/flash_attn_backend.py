
import logging
import os
from functools import lru_cache
from typing import Callable, Optional

import torch

log = logging.getLogger(__name__)

_FA_VERSION_ENV = "FLASH_ATTN_VERSION"


@lru_cache(maxsize=1)
def _detect_gpu_arch() -> str:
    if not torch.cuda.is_available():
        return "default"
    major, _ = torch.cuda.get_device_capability()
    if major >= 100:
        return "blackwell"
    if major >= 9:
        return "hopper"
    return "default"


def _load_fa2() -> Optional[Callable]:
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func
    except (ImportError, ModuleNotFoundError):
        return None


def _load_fa3() -> Optional[Callable]:
    try:
        from flash_attn_3.flash_attn_interface import flash_attn_func
        return flash_attn_func
    except (ImportError, ModuleNotFoundError):
        return None


def _load_fa4() -> Optional[Callable]:
    try:
        from flash_attn.cute.interface import flash_attn_func
        return flash_attn_func
    except (ImportError, ModuleNotFoundError):
        return None


_LOADERS = {"2": _load_fa2, "3": _load_fa3, "4": _load_fa4}
_ARCH_TO_VERSION = {"blackwell": "4", "hopper": "3", "default": "2"}


@lru_cache(maxsize=1)
def get_flash_attn_func() -> Optional[Callable]:
    env_ver = os.environ.get(_FA_VERSION_ENV)
    if env_ver is not None:
        env_ver = env_ver.strip()
        if env_ver not in _LOADERS:
            log.warning(f"{_FA_VERSION_ENV}={env_ver} is invalid (expected 2, 3, or 4), falling back to auto-detect")
            env_ver = None

    if env_ver is not None:
        version = env_ver
        source = f"env {_FA_VERSION_ENV}={env_ver}"
    else:
        arch = _detect_gpu_arch()
        version = _ARCH_TO_VERSION[arch]
        source = f"auto-detect (arch={arch})"

    func = _LOADERS[version]()
    if func is not None:
        log.info(f"Flash Attention: using FA{version} ({source})")
        return func

    for fallback_ver in ["3", "2"]:
        if fallback_ver >= version:
            continue
        func = _LOADERS[fallback_ver]()
        if func is not None:
            log.warning(f"Flash Attention: FA{version} unavailable, falling back to FA{fallback_ver}")
            return func

    log.warning("Flash Attention: no backend available, falling back to torch SDPA")
    return None
