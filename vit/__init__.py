#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata

from .head import HeadConfig
from .head import register_constructors as register_head_constructors
from .vit import ViT, ViTConfig
from .vit import register_constructors as register_vit_constructors


register_vit_constructors()
register_head_constructors()

__version__ = importlib.metadata.version("vit")
__all__ = [
    "ViT",
    "ViTConfig",
    "HeadConfig",
    "register_vit_constructors",
    "register_head_constructors",
]
