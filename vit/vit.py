from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Self, Sequence, Type, cast

import torch
import torch.nn as nn
import yaml
from torch import Tensor

from .head import HeadConfig
from .patch_embed import PatchEmbed2d, PatchEmbed3d
from .pos_enc import LearnablePosition, PositionEncoder
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
    num_register_tokens: int = 0
    pos_enc: PositionEncoder = "fourier"
    qk_bias: bool = False

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

        # Register tokens
        if config.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.empty(config.num_register_tokens, config.hidden_size))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

        self.blocks = nn.ModuleList([self.create_encoder_layer() for _ in range(config.depth)])
        self.output_norm = nn.RMSNorm(config.hidden_size)
        if config.qk_bias:
            self.bias_q = nn.ModuleList(
                [
                    LearnablePosition(config.hidden_size, self.stem.tokenized_size(tuple(config.img_size)))
                    for _ in range(config.depth)
                ]
            )
            self.bias_k = deepcopy(self.bias_q)
        else:
            self.bias_q = self.bias_k = None

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

    def forward(self, x: Tensor, mask: Tensor | None = None, return_register_tokens: bool = False) -> Tensor:
        tokenized_size = self.stem.tokenized_size(x.shape[2:])
        x = self.stem(x)
        if mask is not None:
            x = apply_mask(mask, x)

        B = x.shape[0]
        register_tokens = (
            self.register_tokens.unsqueeze(0).expand(B, -1, -1) if self.register_tokens is not None else None
        )
        x = torch.cat([register_tokens, x], dim=1) if register_tokens is not None else x
        for i, block in enumerate(self.blocks):
            assert isinstance(block, TransformerEncoderLayer)
            if self.bias_q is not None and self.bias_k is not None:
                bias_q = self.bias_q[i](tokenized_size).expand(B, -1, -1) # type: ignore
                bias_k = self.bias_k[i](tokenized_size).expand(B, -1, -1) # type: ignore
                if mask is not None:
                    bias_q = apply_mask(mask, bias_q)
                    bias_k = apply_mask(mask, bias_k)
                if register_tokens is not None:
                    bias_q = torch.cat([torch.zeros_like(register_tokens), bias_q], dim=1)
                    bias_k = torch.cat([torch.zeros_like(register_tokens), bias_k], dim=1)
            else:
                bias_q = bias_k = None

            x = block(x, bias_q=bias_q, bias_k=bias_k)

        if self.output_norm is not None:
            x = self.output_norm(x)

        if return_register_tokens:
            return x
        else:
            return x[..., self.config.num_register_tokens :, :].contiguous()

    def mlp_requires_grad_(self, requires_grad: bool = True) -> None:
        for block in self.blocks:
            layer = cast(nn.Module, block.mlp)
            layer.requires_grad_(requires_grad)

    def self_attention_requires_grad_(self, requires_grad: bool = True) -> None:
        for block in self.blocks:
            layer = cast(nn.Module, block.self_attention)
            layer.requires_grad_(requires_grad)


register_constructors()
