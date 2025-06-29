from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Self, Sequence, Type, cast

import torch
import torch.nn as nn
import yaml
from torch import Tensor

from .head import HeadConfig
from .patch_embed import PatchEmbed2d, PatchEmbed3d
from .pos_enc import PositionEncoder
from .tokens import apply_mask, create_mask
from .transformer import CrossAttentionTransformer, TransformerDecoderLayer, TransformerEncoderLayer


def vit_config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return ViTConfig(**values)


def register_constructors():
    tags = [
        "tag:yaml.org,2002:python/object:vit.vit.ViTConfig",
        "tag:yaml.org,2002:python/object:vit.ViTConfig",
    ]
    loaders = [yaml.SafeLoader, yaml.FullLoader, yaml.UnsafeLoader]
    for tag in tags:
        for loader in loaders:
            loader.add_constructor(tag, vit_config_constructor)


@dataclass(frozen=True)
class ViTConfig:
    # Inputs
    in_channels: int
    patch_size: Sequence[int]
    img_size: Sequence[int]

    # Transformer
    depth: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    bias: bool = True
    activation: str = "srelu"
    drop_path_rate: float = 0.0
    pos_enc: PositionEncoder = "fourier"
    layer_scale: float | None = None

    # Special tokens
    special_tokens: Sequence[str] = field(default_factory=lambda: ["[CLS]"])
    num_register_tokens: int = 0

    # Trainable blocks
    mlp_requires_grad: bool = True
    self_attention_requires_grad: bool = True

    # Heads
    heads: Dict[str, HeadConfig] = field(default_factory=dict)

    def instantiate(self) -> "ViT":
        return ViT(self)

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


@torch.compile(fullgraph=True, dynamic=False)
def _prepend_special_tokens(special_tokens: nn.ParameterDict, x: Tensor) -> Tensor:
    B, _, D = x.shape
    _special_tokens = torch.stack([special_tokens[token] for token in special_tokens.keys()], dim=0)
    return torch.cat([_special_tokens.view(1, len(special_tokens), D).expand(B, -1, -1), x], dim=1)


@torch.compile(fullgraph=True, dynamic=False)
def _dense_result_to_dict(special_tokens: nn.ParameterDict, x: Tensor, x_pre_norm: Tensor) -> Dict[str, Tensor]:
    result: Dict[str, Tensor] = {}
    for i, k in enumerate(special_tokens.keys()):
        result[k] = x[..., i, :].contiguous()
    result["features"] = x[..., len(special_tokens) :, :].contiguous()
    result["pre_norm"] = x_pre_norm
    return result


class ViT(nn.Module):

    def __init__(self, config: ViTConfig):
        super().__init__()
        self._config = config

        # Stem tokenizer
        PatchEmbed = PatchEmbed2d if len(config.patch_size) == 2 else PatchEmbed3d
        self.stem = PatchEmbed(
            config.in_channels,
            config.hidden_size,
            config.patch_size,
            config.img_size,
            pos_enc=config.pos_enc,
        )

        # Special tokens
        self.special_tokens = nn.ParameterDict()
        for token in config.special_tokens:
            self.special_tokens[token] = nn.Parameter(torch.empty(config.hidden_size))
            nn.init.normal_(self.special_tokens[token], std=1e-6)
        for i in range(config.num_register_tokens):
            self.special_tokens[f"[REG_{i}]"] = nn.Parameter(torch.empty(config.hidden_size))
            nn.init.normal_(self.special_tokens[f"[REG_{i}]"], std=1e-6)

        self.blocks = nn.ModuleList([self.create_encoder_layer() for _ in range(config.depth)])
        self.output_norm = nn.RMSNorm(config.hidden_size)

        self.mlp_requires_grad_(self.config.mlp_requires_grad)
        self.self_attention_requires_grad_(self.config.self_attention_requires_grad)

        self.heads = nn.ModuleDict(
            {name: head_config.instantiate(config) for name, head_config in config.heads.items()}
        )

    @property
    def config(self) -> ViTConfig:
        return self._config

    def create_encoder_layer(self) -> TransformerEncoderLayer:
        return TransformerEncoderLayer(
            self.config.hidden_size,
            self.config.ffn_hidden_size,
            self.config.num_attention_heads,
            self.config.hidden_dropout,
            self.config.attention_dropout,
            self.config.bias,
            self.config.activation,
            self.config.drop_path_rate,
            layer_scale=self.config.layer_scale,
        )

    def create_decoder_layer(self) -> TransformerDecoderLayer:
        return TransformerDecoderLayer(
            self.config.hidden_size,
            self.config.ffn_hidden_size,
            self.config.num_attention_heads,
            self.config.hidden_dropout,
            self.config.attention_dropout,
            self.config.bias,
            self.config.activation,
            self.config.drop_path_rate,
            layer_scale=self.config.layer_scale,
        )

    def create_cross_attention_layer(self) -> CrossAttentionTransformer:
        return CrossAttentionTransformer(
            self.config.hidden_size,
            self.config.ffn_hidden_size,
            self.config.num_attention_heads,
            self.config.hidden_dropout,
            self.config.attention_dropout,
            self.config.bias,
            self.config.activation,
            self.config.drop_path_rate,
            layer_scale=self.config.layer_scale,
        )

    def create_mask(
        self,
        input: Tensor,
        unmasked_ratio: float,
        scale: int,
        roll: bool = False,
    ) -> Tensor:
        r"""Creates a token mask for the input.

        Args:
            input: Input tensor from which to infer mask properties.
                Should be a raw input prior to tokenization.
            unmasked_ratio: Proportion of tokens to leave unmasked.
            scale: Scale of the mask.
            roll: Whether to roll the mask.

        Shapes:
            - input: :math:`(B, C, H, W)` or :math:`(B, C, D, H, W)`
            - output: :math:`(B, L)`

        Returns:
            Token mask.
        """
        batch_size = input.shape[0]
        device = input.device
        original_size = input.shape[2:]
        tokenized_size = self.stem.tokenized_size(cast(Any, original_size))
        mask = create_mask(
            tokenized_size,
            mask_ratio=1 - unmasked_ratio,
            batch_size=batch_size,
            scale=scale,
            roll=roll,
            device=device,
        )

        return mask

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Dict[str, Tensor]:
        x = self.stem(x)
        if mask is not None:
            x = apply_mask(mask, x)

        x.shape[0]
        x = _prepend_special_tokens(self.special_tokens, x)
        for block in self.blocks:
            assert isinstance(block, TransformerEncoderLayer)
            x = block(x)

        x_norm = self.output_norm(x)
        return _dense_result_to_dict(self.special_tokens, x_norm, x)

    def mlp_requires_grad_(self, requires_grad: bool = True) -> None:
        for block in self.blocks:
            layer = cast(nn.Module, block.mlp)
            layer.requires_grad_(requires_grad)

    def self_attention_requires_grad_(self, requires_grad: bool = True) -> None:
        for block in self.blocks:
            layer = cast(nn.Module, block.self_attention)
            layer.requires_grad_(requires_grad)


register_constructors()
