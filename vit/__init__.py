#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata

from .head import Head, HeadConfig, TransposedConv2dHead, TransposedConv2dHeadConfig, UpsampleHead, UpsampleHeadConfig
from .head import register_constructors as register_head_constructors
from .vit import ViT, ViTConfig, ViTFeatures
from .vit import register_constructors as register_vit_constructors


register_vit_constructors()
register_head_constructors()

__version__ = importlib.metadata.version("vit")
__all__ = [
    "ViT",
    "ViTConfig",
    "ViTFeatures",
    "HeadConfig",
    "Head",
    "TransposedConv2dHead",
    "TransposedConv2dHeadConfig",
    "UpsampleHead",
    "UpsampleHeadConfig",
    "register_vit_constructors",
    "register_head_constructors",
]
