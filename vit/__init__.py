#!/usr/bin/env python

import importlib
from importlib.metadata import PackageNotFoundError, version

from .attention import AttentivePool
from .head import (
    AttentivePoolHead,
    AttentivePoolHeadConfig,
    Head,
    HeadConfig,
    TransposedConv2dHead,
    TransposedConv2dHeadConfig,
    UpsampleHead,
    UpsampleHeadConfig,
    register_constructors as register_head_constructors,
)
from .vit import ViT, ViTConfig, ViTFeatures, register_constructors as register_vit_constructors


def _resolve_version() -> str:
    try:
        version_module = importlib.import_module("vit._version")
        resolved = getattr(version_module, "__version__", None)
        if isinstance(resolved, str):
            return resolved
    except ModuleNotFoundError:
        pass

    try:
        return version("vit")
    except PackageNotFoundError:
        return "0+unknown"


__version__: str = _resolve_version()

register_vit_constructors()
register_head_constructors()
__all__ = [
    "__version__",
    "ViT",
    "ViTConfig",
    "ViTFeatures",
    "AttentivePool",
    "AttentivePoolHead",
    "AttentivePoolHeadConfig",
    "HeadConfig",
    "Head",
    "TransposedConv2dHead",
    "TransposedConv2dHeadConfig",
    "UpsampleHead",
    "UpsampleHeadConfig",
    "register_vit_constructors",
    "register_head_constructors",
]
