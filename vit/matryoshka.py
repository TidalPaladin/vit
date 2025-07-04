from dataclasses import dataclass
from pathlib import Path
from typing import Self, Type

import yaml


@dataclass
class MatryoshkaConfig:
    ffn_size: int
    skip_every_n_layers: int = 0

    def __post_init__(self):
        if not 0 < self.ffn_size:
            raise ValueError("ffn_size must be positive")  # pragma: no cover
        if self.skip_every_n_layers != 0 and self.skip_every_n_layers < 2:
            raise ValueError("skip_every_n_layers must be 0 or greater than 1")  # pragma: no cover

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


def matryoshka_config_constructor(loader, node):
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
            loader.add_constructor(tag, matryoshka_config_constructor)
