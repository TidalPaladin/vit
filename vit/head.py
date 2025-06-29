from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch.nn as nn
import yaml


if TYPE_CHECKING:
    from .vit import ViTConfig
else:
    ViTConfig = Any


def head_config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return HeadConfig(**values)


def register_constructors():
    tags = [
        "tag:yaml.org,2002:python/object:vit.head.HeadConfig",
        "tag:yaml.org,2002:python/object:vit.HeadConfig",
    ]
    loaders = [yaml.SafeLoader, yaml.FullLoader, yaml.UnsafeLoader]
    for tag in tags:
        for loader in loaders:
            loader.add_constructor(tag, head_config_constructor)


@dataclass
class HeadConfig:
    key: str = "[CLS]"
    out_dim: int | None = None
    stop_gradient: bool = False

    def instantiate(self, backbone_config: ViTConfig) -> nn.Linear:
        return nn.Linear(backbone_config.hidden_size, self.out_dim or backbone_config.hidden_size)
