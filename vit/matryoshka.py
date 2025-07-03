from dataclasses import dataclass
from pathlib import Path
from typing import Self, Type

import torch
import yaml
from torch import Tensor


@torch.compile(fullgraph=True, dynamic=False)
def slice_matryoshka(x: Tensor, frac: float = 1.0) -> Tensor:
    D = x.shape[-1]
    D_sliced = int(D * frac)
    return x[..., :D_sliced]


@torch.compile(fullgraph=True, dynamic=False)
def slice_matryoshka_weight(w: Tensor, input_frac: float = 1.0, output_frac: float = 1.0) -> Tensor:
    D_out, D_in = w.shape
    D_in_sliced = int(D_in * input_frac)
    D_out_sliced = int(D_out * output_frac)
    return w[..., :D_out_sliced, :D_in_sliced]


@torch.compile(fullgraph=True, dynamic=False)
def slice_matryoshka_heads(x: Tensor, frac: float = 1.0) -> Tensor:
    _, H, _, _ = x.shape
    H_sliced = int(H * frac)
    return x[..., :H_sliced, :, :]


@torch.compile(fullgraph=True, dynamic=False)
def unslice_matryoshka(x: Tensor, size: int) -> Tensor:
    if size == x.shape[-1]:
        return x
    out = x.new_zeros(*x.shape[:-1], size)
    out[..., : x.shape[-1]] = x
    return out


@dataclass
class MatryoshkaConfig:
    feature_frac: float = 1.0
    feedforward_frac: float = 1.0
    heads_frac: float = 1.0
    depth_stride: int = 1

    def __post_init__(self):
        if not 0 < self.feature_frac <= 1:
            raise ValueError("feature_frac must be between 0 and 1")  # pragma: no cover
        if not 0 < self.feedforward_frac <= 1:
            raise ValueError("feedforward_frac must be between 0 and 1")  # pragma: no cover
        if not 0 < self.heads_frac <= 1:
            raise ValueError("heads_frac must be between 0 and 1")  # pragma: no cover
        if not 0 < self.depth_stride:
            raise ValueError("depth_stride must be positive")  # pragma: no cover

    @classmethod
    def from_yaml(cls: Type[Self], path: str | Path) -> Self:
        if isinstance(path, Path):
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, "r") as f:
                config = yaml.full_load(f)
            return cls(**config)

        elif isinstance(path, str) and path.endswith(".yaml"):
            return cls.from_yaml(Path(path))

        else:
            config = yaml.full_load(path)
            return cls(**config)

    def to_yaml(self) -> str:
        return yaml.dump(self.__dict__)


def vit_config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return MatryoshkaConfig(**values)


def register_constructors():
    tags = [
        "tag:yaml.org,2002:python/object:vit.matryoshka.MatryoshkaConfig",
        "tag:yaml.org,2002:python/object:vit.MatryoshkaConfig",
    ]
    loaders = [yaml.SafeLoader, yaml.FullLoader, yaml.UnsafeLoader]
    for tag in tags:
        for loader in loaders:
            loader.add_constructor(tag, vit_config_constructor)
