#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata

from vit.vit import ViT, ViTConfig


__version__ = importlib.metadata.version("vit")
__all__ = ["ViT", "ViTConfig"]
