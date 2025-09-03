from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Self, Sequence, Type, cast

import torch
import torch.nn as nn
import yaml
from torch import Tensor

from .head import Head, HeadConfig, MLPHead
from .patch_embed import PatchEmbed2d, PatchEmbed3d
from .pos_enc import PositionEncoder
from .rope import RopePositionEmbedding
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
    attention_bias: bool = True
    mlp_bias: bool = True
    activation: str = "srelu"
    drop_path_rate: float = 0.0
    num_register_tokens: int = 0
    pos_enc: PositionEncoder = "fourier"
    layer_scale: float | None = None
    glu_limit: float | None = None
    glu_extra_bias: float | None = None

    # RoPE options
    rope_normalize_coords: Literal["min", "max", "separate"] = "separate"
    rope_base: float = 100
    rope_shift_coords: float | None = None
    rope_jitter_coords: float | None = None
    rope_rescale_coords: float | None = None

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
            pos_enc=config.pos_enc if config.pos_enc != "rope" else "none",
        )

        if config.pos_enc == "rope":
            self.rope = RopePositionEmbedding(
                config.hidden_size,
                base=config.rope_base,
                num_heads=config.num_attention_heads,
                rescale_coords=config.rope_rescale_coords,
                shift_coords=config.rope_shift_coords,
                jitter_coords=config.rope_jitter_coords,
            )
        else:
            self.rope = None

        # Register tokens
        self.register_tokens = nn.Parameter(
            torch.empty(1, config.num_register_tokens, config.hidden_size),
            requires_grad=config.num_register_tokens > 0,
        )
        nn.init.normal_(self.register_tokens, std=0.02)

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
            self.config.attention_bias,
            self.config.mlp_bias,
            self.config.activation,
            self.config.drop_path_rate,
            layer_scale=self.config.layer_scale,
            glu_limit=self.config.glu_limit,
            glu_extra_bias=self.config.glu_extra_bias,
        )

    def create_decoder_layer(self) -> TransformerDecoderLayer:
        return TransformerDecoderLayer(
            self.config.hidden_size,
            self.config.ffn_hidden_size,
            self.config.num_attention_heads,
            self.config.hidden_dropout,
            self.config.attention_dropout,
            self.config.attention_bias,
            self.config.mlp_bias,
            self.config.activation,
            self.config.drop_path_rate,
            layer_scale=self.config.layer_scale,
            glu_limit=self.config.glu_limit,
            glu_extra_bias=self.config.glu_extra_bias,
        )

    def create_cross_attention_layer(self) -> CrossAttentionTransformer:
        return CrossAttentionTransformer(
            self.config.hidden_size,
            self.config.ffn_hidden_size,
            self.config.num_attention_heads,
            self.config.hidden_dropout,
            self.config.attention_dropout,
            self.config.attention_bias,
            self.config.mlp_bias,
            self.config.activation,
            self.config.drop_path_rate,
            layer_scale=self.config.layer_scale,
            glu_limit=self.config.glu_limit,
            glu_extra_bias=self.config.glu_extra_bias,
        )

    def get_head(self, name: str) -> Head | MLPHead:
        head = self.heads[name]
        assert isinstance(head, Head | MLPHead)
        return head

    def get_block(self, i: int) -> TransformerEncoderLayer:
        block = self.blocks[i]
        assert isinstance(block, TransformerEncoderLayer)
        return block

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

    @property
    def prefix_length(self) -> int:
        return self.config.num_register_tokens

    @torch.compile(fullgraph=True)
    def add_prefix_tokens(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        register_tokens = self.register_tokens.expand(B, -1, -1)
        return torch.cat([register_tokens, x], dim=1)

    @torch.compile(fullgraph=True)
    def separate_prefix_tokens(self, x: Tensor) -> tuple[Tensor, Tensor]:
        prefix, x = x.split([self.prefix_length, x.shape[1] - self.prefix_length], dim=1)
        return prefix, x

    def prepare_rope(
        self,
        tokenized_size: Sequence[int],
        mask: Tensor | None = None,
        rope_seed: int | None = None,
    ) -> Tensor:
        if self.rope is None:
            raise ValueError("RoPE is not enabled")

        if len(tokenized_size) == 2:
            H, W = tokenized_size
            rope = self.rope(H=H, W=W, rope_seed=rope_seed)
        else:
            raise ValueError(f"RoPE not supported for non-2D input, got {tokenized_size}")

        if mask is not None:
            B = mask.shape[0]
            sin, cos = rope
            sin = apply_mask(mask, sin[None].expand(B, -1, -1))
            cos = apply_mask(mask, cos[None].expand(B, -1, -1))
            rope = torch.stack([sin[:, None, ...], cos[:, None, ...]], dim=0)

        return rope

    def forward(
        self, x: Tensor, mask: Tensor | None = None, return_register_tokens: bool = False, rope_seed: int | None = None
    ) -> Tensor:
        # Prepare transformer input
        tokenized_size = self.stem.tokenized_size(x.shape[2:])
        x = self.stem(x)
        x = apply_mask(mask, x) if mask is not None else x
        x = self.add_prefix_tokens(x)

        # Prepare RoPE sin/cos if needed
        rope = self.prepare_rope(tokenized_size, mask, rope_seed) if self.rope is not None else None

        # Apply transformer
        for block in self.blocks:
            assert isinstance(block, TransformerEncoderLayer)
            x = block(x, rope=rope)

        # Prepare output
        x = self.output_norm(x)
        if not return_register_tokens:
            _, x = self.separate_prefix_tokens(x)
        return x

    if TYPE_CHECKING:

        def __call__(
            self,
            x: Tensor,
            mask: Tensor | None = None,
            return_register_tokens: bool = False,
            rope_seed: int | None = None,
        ) -> Tensor:
            return self.forward(x, mask, return_register_tokens, rope_seed)

    @torch.no_grad()
    def _reshape_attention_weights(self, w: Tensor, tokenized_size: Sequence[int]) -> Tensor:
        B, H, Lq, Lk = w.shape
        assert Lq == Lk, f"Query and key lengths must match, got {Lq} and {Lk}"
        w = w[..., self.config.num_register_tokens :].view(B, H, Lq, *tokenized_size)
        return w.contiguous()

    def forward_attention_weights(self, x: Tensor) -> Dict[str, Tensor]:
        # Prepare transformer input
        tokenized_size = self.stem.tokenized_size(x.shape[2:])
        x = self.stem(x)
        x = self.add_prefix_tokens(x)

        # Apply transformer
        weights: Dict[str, Tensor] = {}
        for i, block in enumerate(self.blocks):
            assert isinstance(block, TransformerEncoderLayer)
            w_i = block.self_attention.forward_weights(x)
            weights[f"layer_{i}"] = self._reshape_attention_weights(w_i, tokenized_size)
            x = block(x)

        return weights

    def mlp_requires_grad_(self, requires_grad: bool = True) -> None:
        for block in self.blocks:
            layer = cast(nn.Module, block.mlp)
            layer.requires_grad_(requires_grad)

    def self_attention_requires_grad_(self, requires_grad: bool = True) -> None:
        for block in self.blocks:
            layer = cast(nn.Module, block.self_attention)
            layer.requires_grad_(requires_grad)

    def backbone_requires_grad_(self, requires_grad: bool = True) -> None:
        self.stem.requires_grad_(requires_grad)
        self.blocks.requires_grad_(requires_grad)
        self.output_norm.requires_grad_(requires_grad)
        if self.register_tokens is not None:
            self.register_tokens.requires_grad_(requires_grad)


register_constructors()
