import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.utils import _pytree
from torch.utils.checkpoint import checkpoint

from .attention import (
    TokenSpecializedAttentionCompileMode,
    _validate_token_specialized_attention_compile_policy,
)
from .fused import validate_adaln_gate_init
from .head import (
    AttentivePoolHead,
    AttentivePoolHeadConfig,
    Head,
    HeadConfig,
    TransposedConv2dHead,
    TransposedConv2dHeadConfig,
    UpsampleHead,
    UpsampleHeadConfig,
)
from .initialization import trunc_normal_
from .norm import NORM_TYPE_CHOICES, NormType, make_norm
from .packed import PackedAttentionBackend, PackedSequence
from .patch_embed import PatchEmbed2d, PatchEmbed3d
from .pos_enc import PositionEncoder
from .rope import RopePositionEmbedding
from .tokens import apply_mask, create_mask
from .transformer import CrossAttentionTransformer, TransformerDecoderLayer, TransformerEncoderLayer


HeadConfigType = HeadConfig | AttentivePoolHeadConfig | TransposedConv2dHeadConfig | UpsampleHeadConfig
HeadModuleType = Head | AttentivePoolHead | TransposedConv2dHead | UpsampleHead
_EXPLAINABILITY_TRACE_ACTIVE = ContextVar("vit_explainability_trace_active", default=False)
_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype representation to torch.dtype."""
    try:
        return _DTYPE_BY_NAME[dtype_str]
    except KeyError:
        supported_dtypes = ", ".join(sorted(_DTYPE_BY_NAME))
        raise ValueError(f"Unsupported dtype {dtype_str!r}. Expected one of: {supported_dtypes}") from None


def vit_config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    # Convert dtype string to torch.dtype
    if "dtype" in values and isinstance(values["dtype"], str):
        values["dtype"] = _parse_dtype(values["dtype"])
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
    """Configuration for ViT construction.

    When enabling conditioned MLPs via `conditioning_size`, keep `adaln_gate_init=0.0`
    for the default AdaLN-Zero initialization. Set `adaln_gate_init=1.0` when
    converting a pretrained unconditioned MLP stack into a conditioned one so the
    loaded MLP path is preserved at initialization.

    Set `specialize_global_token_norms=True` to give the CLS/register prefix and
    visual tokens independent pre-attention and pre-MLP norms. Configured LayerScale
    parameters are separated with them. Set `specialize_global_token_qkv_blocks` to
    the number of leading blocks that also receive independent QKV projections. Both
    options are disabled by default, preserving the shared-token-path architecture.
    `token_specialized_attention_compile_mode` selects the runtime compiler policy;
    `token_specialized_attention_static_batch_sizes` optionally bounds the batches
    routed to or accepted by concrete-shape modes.
    """

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
    num_cls_tokens: int = 0
    pos_enc: PositionEncoder = "rope"
    layer_scale: float | None = None
    glu_limit: float | None = None
    glu_extra_bias: float | None = None
    glu_max_autotune_gemm: bool = False

    # RoPE options
    rope_normalize_coords: Literal["min", "max", "separate"] = "separate"
    rope_base: float = 100
    rope_shift_coords: float | None = None
    rope_jitter_coords: float | None = None
    rope_rescale_coords: float | None = None

    # Trainable blocks
    mlp_requires_grad: bool = True
    self_attention_requires_grad: bool = True

    # Memory optimization
    activation_checkpointing: bool = False
    conditioning_size: int | None = None
    adaln_gate_init: float = 0.0

    # Master weight dtype (default BF16)
    dtype: torch.dtype = torch.bfloat16
    norm_type: NormType = "rmsnorm"
    qk_normalization: bool = False
    patch_embed_normalization: bool = False

    # Heads
    heads: dict[str, HeadConfigType] = field(default_factory=dict)

    # Global and visual token pathways
    specialize_global_token_norms: bool = False
    specialize_global_token_qkv_blocks: int = 0
    token_specialized_attention_compile_mode: TokenSpecializedAttentionCompileMode = "auto"
    token_specialized_attention_static_batch_sizes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.norm_type not in NORM_TYPE_CHOICES:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")
        if self.pos_enc == "fourier" and self.hidden_size % 2 != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be even when using Fourier positional encoding")
        if self.conditioning_size is not None and self.conditioning_size <= 0:
            raise ValueError(f"conditioning_size must be positive when provided, got {self.conditioning_size}")
        if self.specialize_global_token_qkv_blocks < 0:
            raise ValueError("specialize_global_token_qkv_blocks must be non-negative")
        if self.specialize_global_token_qkv_blocks > self.depth:
            raise ValueError("specialize_global_token_qkv_blocks cannot exceed depth")
        if self.token_specialization_enabled and self.num_global_tokens == 0:
            raise ValueError("global-token specialization requires at least one CLS or register token")
        if self.specialize_global_token_norms and self.conditioning_size is not None:
            raise ValueError("global-token normalization specialization is incompatible with conditioned MLPs")
        if self.glu_max_autotune_gemm and not self.activation.endswith("glu"):
            raise ValueError("glu_max_autotune_gemm requires a GLU activation")
        normalized_static_batch_sizes = _validate_token_specialized_attention_compile_policy(
            self.token_specialized_attention_compile_mode,
            self.token_specialized_attention_static_batch_sizes,
            specialization_enabled=self.token_specialization_enabled,
        )
        object.__setattr__(
            self,
            "token_specialized_attention_static_batch_sizes",
            normalized_static_batch_sizes,
        )
        validate_adaln_gate_init(self.adaln_gate_init)

    @property
    def num_global_tokens(self) -> int:
        """Return the CLS/register prefix length used by token specialization."""
        return self.num_cls_tokens + self.num_register_tokens

    @property
    def token_specialization_enabled(self) -> bool:
        """Return whether any global/visual token pathway is separated."""
        return self.specialize_global_token_norms or self.specialize_global_token_qkv_blocks > 0

    def instantiate(self, device: torch.device | None = None) -> "ViT":
        return ViT(self, device=device)

    @classmethod
    def from_yaml(cls: type[Self], path: str | Path) -> Self:
        if isinstance(path, Path):
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            with open(path) as f:
                config = yaml.full_load(f)
        elif isinstance(path, str) and path.endswith(".yaml"):
            return cls.from_yaml(Path(path))
        else:
            config = yaml.full_load(path)

        # Convert dtype string to torch.dtype
        if "dtype" in config and isinstance(config["dtype"], str):
            config["dtype"] = _parse_dtype(config["dtype"])
        return cls(**config)

    def to_yaml(self) -> str:
        # Convert dtype to string for YAML serialization
        data = {**self.__dict__}
        if "dtype" in data and isinstance(data["dtype"], torch.dtype):
            data["dtype"] = str(data["dtype"]).replace("torch.", "")
        return yaml.dump(data)


class ViTFeatures:
    def __init__(
        self,
        dense_features: Tensor,
        num_register_tokens: int,
        num_cls_tokens: int,
        tokenized_size: Sequence[int] | None = None,
    ):
        self._dense_features = dense_features
        self._num_register_tokens = num_register_tokens
        self._num_cls_tokens = num_cls_tokens
        self._tokenized_size = tuple(tokenized_size) if tokenized_size is not None else None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"cls_tokens={tuple(self.cls_tokens.shape)}, "
            f"register_tokens={tuple(self.register_tokens.shape)}, "
            f"visual_tokens={tuple(self.visual_tokens.shape)})"
        )

    def __iter__(self) -> Iterator[Tensor]:
        yield self.cls_tokens
        yield self.register_tokens
        yield self.visual_tokens

    @property
    def dense_features(self) -> Tensor:
        return self._dense_features

    @property
    def num_register_tokens(self) -> int:
        return self._num_register_tokens

    @property
    def num_cls_tokens(self) -> int:
        return self._num_cls_tokens

    @property
    def visual_tokens(self) -> Tensor:
        start = self.num_register_tokens + self.num_cls_tokens
        return self.dense_features[..., start:, :]

    @property
    def register_tokens(self) -> Tensor:
        start = self.num_cls_tokens
        end = self.num_cls_tokens + self.num_register_tokens
        return self.dense_features[..., start:end, :]

    @property
    def cls_tokens(self) -> Tensor:
        end = self.num_cls_tokens
        return self.dense_features[..., :end, :]

    @property
    def tokenized_size(self) -> tuple[int, ...] | None:
        return self._tokenized_size

    @property
    def visual_tokens_as_grid(self) -> Tensor:
        """Returns visual tokens reshaped to spatial grid.

        Shapes:
            - For 2D: (B, L, C) -> (B, H, W, C)
            - For 3D: (B, L, C) -> (B, D, H, W, C)

        Returns:
            Reshaped visual tokens with spatial dimensions.

        Raises:
            ValueError: If tokenized_size is not set.
        """
        if self._tokenized_size is None:
            raise ValueError("tokenized_size is not set, cannot reshape to grid")
        visual = self.visual_tokens
        B, L, C = visual.shape
        return visual.view(B, *self._tokenized_size, C)

    def apply(self: Self, func: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(
            func(self.dense_features), self.num_register_tokens, self.num_cls_tokens, self._tokenized_size
        )

    @classmethod
    def from_separate_features(
        cls: type[Self],
        cls_tokens: Tensor,
        register_tokens: Tensor,
        visual_tokens: Tensor,
        tokenized_size: Sequence[int] | None = None,
    ) -> Self:
        return cls(
            dense_features=torch.cat([cls_tokens, register_tokens, visual_tokens], dim=1),
            num_register_tokens=register_tokens.shape[1],
            num_cls_tokens=cls_tokens.shape[1],
            tokenized_size=tokenized_size,
        )


class PackedViTFeatures:
    """ViT features with fixed prefixes kept dense and visual tokens kept packed."""

    def __init__(
        self,
        cls_tokens: Tensor,
        register_tokens: Tensor,
        visual_tokens: PackedSequence,
        tokenized_size: Sequence[int] | None = None,
    ):
        if cls_tokens.ndim != 3 or register_tokens.ndim != 3:
            raise ValueError("packed ViT prefix tokens must be dense rank-three tensors")
        if cls_tokens.shape[0] != visual_tokens.batch_size or register_tokens.shape[0] != visual_tokens.batch_size:
            raise ValueError("packed ViT prefix and visual batch sizes must match")
        if cls_tokens.device != visual_tokens.values.device or register_tokens.device != visual_tokens.values.device:
            raise ValueError("packed ViT prefix and visual tokens must use the same device")
        if (
            cls_tokens.shape[-1] != visual_tokens.values.shape[-1]
            or register_tokens.shape[-1] != visual_tokens.values.shape[-1]
        ):
            raise ValueError("packed ViT prefix and visual hidden sizes must match")
        self._cls_tokens = cls_tokens
        self._register_tokens = register_tokens
        self._visual_tokens = visual_tokens
        self._tokenized_size = tuple(tokenized_size) if tokenized_size is not None else None

    @property
    def cls_tokens(self) -> Tensor:
        return self._cls_tokens

    @property
    def register_tokens(self) -> Tensor:
        return self._register_tokens

    @property
    def visual_tokens(self) -> PackedSequence:
        return self._visual_tokens

    @property
    def tokenized_size(self) -> tuple[int, ...] | None:
        return self._tokenized_size

    def to_padded(self, padding_value: float = 0.0) -> tuple[ViTFeatures, Tensor]:
        """Explicitly convert to legacy dense features and return visual validity."""
        visual_tokens, validity = self.visual_tokens.to_padded(padding_value)
        tokenized_size = self.tokenized_size
        if tokenized_size is not None and not bool((self.visual_tokens.lengths == math.prod(tokenized_size)).all()):
            tokenized_size = None
        return (
            ViTFeatures.from_separate_features(
                self.cls_tokens,
                self.register_tokens,
                visual_tokens,
                tokenized_size,
            ),
            validity,
        )


def _flatten_vit_features(features: ViTFeatures) -> tuple[list[Tensor], list[int | list[int] | None]]:
    tokenized_size = list(features.tokenized_size) if features.tokenized_size is not None else None
    context: list[int | list[int] | None] = [
        features.num_register_tokens,
        features.num_cls_tokens,
        tokenized_size,
    ]
    return [features.dense_features], context


def _unflatten_vit_features(
    values: Iterable[Tensor],
    context: list[int | list[int] | None],
) -> ViTFeatures:
    (dense_features,) = values
    num_register_tokens, num_cls_tokens, tokenized_size = context
    assert isinstance(num_register_tokens, int)
    assert isinstance(num_cls_tokens, int)
    assert tokenized_size is None or isinstance(tokenized_size, list)
    return ViTFeatures(dense_features, num_register_tokens, num_cls_tokens, tokenized_size)


_pytree.register_pytree_node(
    ViTFeatures,
    _flatten_vit_features,
    _unflatten_vit_features,
    serialized_type_name="vit.ViTFeatures",
)


def _packed_visual_destinations(
    visual: PackedSequence,
    prefix_length: int,
) -> tuple[Tensor, Tensor, Tensor]:
    new_lengths = visual.lengths + prefix_length
    zero = torch.zeros(1, device=visual.values.device, dtype=torch.int32)
    new_offsets = torch.cat((zero, new_lengths.cumsum(0, dtype=torch.int32)))
    sequence_indices = torch.repeat_interleave(
        torch.arange(visual.batch_size, device=visual.values.device),
        visual.lengths.to(torch.int64),
    )
    source_starts = torch.repeat_interleave(
        visual._cu_seqlens_for_ops()[:-1].to(torch.int64),
        visual.lengths.to(torch.int64),
    )
    within_sequence = torch.arange(visual.values.shape[0], device=visual.values.device) - source_starts
    destinations = new_offsets[:-1].to(torch.int64).index_select(0, sequence_indices)
    destinations = destinations + prefix_length + within_sequence
    return destinations, new_offsets, new_lengths


def _add_packed_prefix(
    visual: PackedSequence,
    cls_tokens: Tensor,
    register_tokens: Tensor,
    visual_rope: Tensor | None,
) -> tuple[PackedSequence, Tensor | None]:
    prefix = torch.cat(
        (cls_tokens.expand(visual.batch_size, -1, -1), register_tokens.expand(visual.batch_size, -1, -1)), dim=1
    )
    prefix_length = prefix.shape[1]
    if prefix_length == 0:
        return visual, visual_rope

    visual_destinations, new_offsets, new_lengths = _packed_visual_destinations(visual, prefix_length)
    total_tokens = visual.values.shape[0] + visual.batch_size * prefix_length
    output = visual.values.new_zeros((total_tokens, visual.values.shape[-1]))
    output = output.index_copy(0, visual_destinations, visual.values)
    prefix_destinations = (
        new_offsets[:-1].to(torch.int64).unsqueeze(1)
        + torch.arange(prefix_length, device=visual.values.device).unsqueeze(0)
    ).flatten()
    output = output.index_copy(0, prefix_destinations, prefix.reshape(-1, prefix.shape[-1]))
    packed = PackedSequence._from_validated(
        output,
        new_offsets,
        new_lengths,
        visual.min_seqlen + prefix_length,
        visual.max_seqlen + prefix_length,
    )

    if visual_rope is None:
        return packed, None
    if visual_rope.shape[1] != visual.values.shape[0]:
        raise ValueError("packed visual RoPE must align one-to-one with visual token values")
    full_rope = visual_rope.new_empty((2, total_tokens, visual_rope.shape[-1]))
    full_rope[0].zero_()
    full_rope[1].fill_(1)
    full_rope = full_rope.index_copy(1, visual_destinations, visual_rope)
    return packed, full_rope


def _split_packed_features(
    packed: PackedSequence,
    num_cls_tokens: int,
    num_register_tokens: int,
    tokenized_size: Sequence[int] | None,
) -> PackedViTFeatures:
    prefix_length = num_cls_tokens + num_register_tokens
    sequence_starts = packed._cu_seqlens_for_ops()[:-1].to(torch.int64)
    if prefix_length == 0:
        prefix = packed.values.new_empty((packed.batch_size, 0, packed.values.shape[-1]))
    else:
        prefix_indices = sequence_starts.unsqueeze(1) + torch.arange(prefix_length, device=packed.values.device)
        prefix = packed.values.index_select(0, prefix_indices.flatten()).view(
            packed.batch_size, prefix_length, packed.values.shape[-1]
        )

    visual_lengths = packed.lengths - prefix_length
    source_starts = torch.repeat_interleave(sequence_starts, visual_lengths.to(torch.int64))
    visual_offsets = torch.cat(
        (
            torch.zeros(1, device=packed.values.device, dtype=torch.int32),
            visual_lengths.cumsum(0, dtype=torch.int32),
        )
    )
    total_visual_tokens = packed.values.shape[0] - packed.batch_size * prefix_length
    within_sequence = torch.arange(total_visual_tokens, device=packed.values.device)
    within_sequence -= torch.repeat_interleave(visual_offsets[:-1].to(torch.int64), visual_lengths.to(torch.int64))
    visual_indices = source_starts + prefix_length + within_sequence
    visual = PackedSequence._from_validated(
        packed.values.index_select(0, visual_indices),
        visual_offsets,
        visual_lengths,
        packed.min_seqlen - prefix_length,
        packed.max_seqlen - prefix_length,
    )
    return PackedViTFeatures(
        cls_tokens=prefix[:, :num_cls_tokens],
        register_tokens=prefix[:, num_cls_tokens:],
        visual_tokens=visual,
        tokenized_size=tokenized_size,
    )


class ViT(nn.Module):
    def __init__(
        self,
        config: ViTConfig,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
        device: torch.device | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": config.dtype}
        super().__init__()
        self._config = config
        self._packed_quantization_enabled = any(
            quantization_config is not None
            for quantization_config in (
                mlp_quantization_config,
                qkv_quantization_config,
                attn_quantization_config,
            )
        )

        # Stem tokenizer
        PatchEmbed = PatchEmbed2d if len(config.patch_size) == 2 else PatchEmbed3d
        self.stem = PatchEmbed(
            config.in_channels,
            config.hidden_size,
            config.patch_size,
            config.img_size,
            pos_enc=config.pos_enc if config.pos_enc != "rope" else "none",
            **factory_kwargs,
        )
        self.patch_embed_norm = (
            make_norm(config.hidden_size, config.norm_type, **factory_kwargs)
            if config.patch_embed_normalization
            else None
        )

        if config.pos_enc == "rope":
            self.rope = RopePositionEmbedding(
                config.hidden_size,
                base=config.rope_base,
                num_heads=config.num_attention_heads,
                rescale_coords=config.rope_rescale_coords,
                shift_coords=config.rope_shift_coords,
                jitter_coords=config.rope_jitter_coords,
                **factory_kwargs,
            )
        else:
            self.rope = None

        # Register / CLS tokens
        self.register_tokens = nn.Parameter(
            torch.empty(1, config.num_register_tokens, config.hidden_size, **factory_kwargs),
            requires_grad=config.num_register_tokens > 0,
        )
        self.cls_tokens = nn.Parameter(
            torch.empty(1, config.num_cls_tokens, config.hidden_size, **factory_kwargs),
            requires_grad=config.num_cls_tokens > 0,
        )
        trunc_normal_(self.register_tokens)
        trunc_normal_(self.cls_tokens)

        self.blocks = nn.ModuleList(
            [
                self._create_encoder_block(
                    block_index,
                    mlp_quantization_config,
                    qkv_quantization_config,
                    attn_quantization_config,
                    device,
                )
                for block_index in range(config.depth)
            ]
        )
        self.output_norm = make_norm(config.hidden_size, config.norm_type, device=device, dtype=config.dtype)

        self.mlp_requires_grad_(self.config.mlp_requires_grad)
        self.self_attention_requires_grad_(self.config.self_attention_requires_grad)

        self.heads = nn.ModuleDict(
            {name: self.create_head(name, head_config, device=device) for name, head_config in config.heads.items()}
        )

    def apply_quantization(
        self,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
    ) -> None:
        if any(
            quantization_config is not None
            for quantization_config in (
                mlp_quantization_config,
                qkv_quantization_config,
                attn_quantization_config,
            )
        ):
            self._packed_quantization_enabled = True
        for block in self.blocks:
            assert isinstance(block, TransformerEncoderLayer)
            block.apply_quantization(mlp_quantization_config, qkv_quantization_config, attn_quantization_config)

    @property
    def config(self) -> ViTConfig:
        return self._config

    def _resolve_factory_dtype(self, dtype: torch.dtype | None) -> torch.dtype:
        return self.config.dtype if dtype is None else dtype

    def _create_encoder_block(
        self,
        block_index: int,
        mlp_quantization_config: Any | None,
        qkv_quantization_config: Any | None,
        attn_quantization_config: Any | None,
        device: torch.device | None,
    ) -> TransformerEncoderLayer:
        if self.config.token_specialization_enabled:
            return self.create_encoder_layer(
                mlp_quantization_config,
                qkv_quantization_config,
                attn_quantization_config,
                device,
                block_index=block_index,
            )
        # Keep the historical factory call unchanged for subclasses when the new feature is disabled.
        return self.create_encoder_layer(
            mlp_quantization_config,
            qkv_quantization_config,
            attn_quantization_config,
            device,
        )

    def create_encoder_layer(
        self,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        block_index: int = 0,
    ) -> TransformerEncoderLayer:
        resolved_dtype = self._resolve_factory_dtype(dtype)
        specialize_global_token_qkv = block_index < self.config.specialize_global_token_qkv_blocks
        specialized_attention_enabled = self.config.specialize_global_token_norms or specialize_global_token_qkv
        compile_mode: TokenSpecializedAttentionCompileMode = (
            self.config.token_specialized_attention_compile_mode if specialized_attention_enabled else "auto"
        )
        static_batch_sizes = (
            self.config.token_specialized_attention_static_batch_sizes if specialized_attention_enabled else None
        )
        return TransformerEncoderLayer(
            hidden_size=self.config.hidden_size,
            ffn_hidden_size=self.config.ffn_hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            hidden_dropout=self.config.hidden_dropout,
            attention_dropout=self.config.attention_dropout,
            attention_bias=self.config.attention_bias,
            mlp_bias=self.config.mlp_bias,
            activation=self.config.activation,
            norm_type=self.config.norm_type,
            qk_normalization=self.config.qk_normalization,
            drop_path_rate=self.config.drop_path_rate,
            layer_scale=self.config.layer_scale,
            glu_limit=self.config.glu_limit,
            glu_extra_bias=self.config.glu_extra_bias,
            mlp_quantization_config=mlp_quantization_config,
            qkv_quantization_config=qkv_quantization_config,
            attn_quantization_config=attn_quantization_config,
            device=device,
            dtype=resolved_dtype,
            conditioning_size=self.config.conditioning_size,
            adaln_gate_init=self.config.adaln_gate_init,
            glu_max_autotune_gemm=self.config.glu_max_autotune_gemm,
            num_global_tokens=self.config.num_global_tokens,
            specialize_global_token_norms=self.config.specialize_global_token_norms,
            specialize_global_token_qkv=specialize_global_token_qkv,
            token_specialized_attention_compile_mode=compile_mode,
            token_specialized_attention_static_batch_sizes=static_batch_sizes,
        )

    def create_decoder_layer(
        self,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> TransformerDecoderLayer:
        resolved_dtype = self._resolve_factory_dtype(dtype)
        return TransformerDecoderLayer(
            hidden_size=self.config.hidden_size,
            ffn_hidden_size=self.config.ffn_hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            hidden_dropout=self.config.hidden_dropout,
            attention_dropout=self.config.attention_dropout,
            attention_bias=self.config.attention_bias,
            mlp_bias=self.config.mlp_bias,
            activation=self.config.activation,
            norm_type=self.config.norm_type,
            qk_normalization=self.config.qk_normalization,
            drop_path_rate=self.config.drop_path_rate,
            layer_scale=self.config.layer_scale,
            glu_limit=self.config.glu_limit,
            glu_extra_bias=self.config.glu_extra_bias,
            mlp_quantization_config=mlp_quantization_config,
            qkv_quantization_config=qkv_quantization_config,
            attn_quantization_config=attn_quantization_config,
            device=device,
            dtype=resolved_dtype,
            conditioning_size=self.config.conditioning_size,
            adaln_gate_init=self.config.adaln_gate_init,
            glu_max_autotune_gemm=self.config.glu_max_autotune_gemm,
        )

    def create_cross_attention_layer(
        self,
        mlp_quantization_config: Any | None = None,
        qkv_quantization_config: Any | None = None,
        attn_quantization_config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> CrossAttentionTransformer:
        resolved_dtype = self._resolve_factory_dtype(dtype)
        return CrossAttentionTransformer(
            hidden_size=self.config.hidden_size,
            ffn_hidden_size=self.config.ffn_hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            hidden_dropout=self.config.hidden_dropout,
            attention_dropout=self.config.attention_dropout,
            attention_bias=self.config.attention_bias,
            mlp_bias=self.config.mlp_bias,
            activation=self.config.activation,
            norm_type=self.config.norm_type,
            qk_normalization=self.config.qk_normalization,
            drop_path_rate=self.config.drop_path_rate,
            layer_scale=self.config.layer_scale,
            glu_limit=self.config.glu_limit,
            glu_extra_bias=self.config.glu_extra_bias,
            mlp_quantization_config=mlp_quantization_config,
            qkv_quantization_config=qkv_quantization_config,
            attn_quantization_config=attn_quantization_config,
            device=device,
            dtype=resolved_dtype,
            conditioning_size=self.config.conditioning_size,
            adaln_gate_init=self.config.adaln_gate_init,
            glu_max_autotune_gemm=self.config.glu_max_autotune_gemm,
        )

    def create_head(
        self,
        name: str,
        head_config: HeadConfigType,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> HeadModuleType:
        _ = name
        resolved_dtype = self._resolve_factory_dtype(dtype)
        head = head_config.instantiate(self.config, device=device, dtype=resolved_dtype)
        assert isinstance(head, HeadModuleType)
        return head

    def get_head(self, name: str) -> HeadModuleType:
        head = self.heads[name]
        assert isinstance(head, HeadModuleType)
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
        return self.config.num_cls_tokens + self.config.num_register_tokens

    @torch.compile(fullgraph=True)
    def add_prefix_tokens(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        register_tokens = self.register_tokens.expand(B, -1, -1)
        cls_tokens = self.cls_tokens.expand(B, -1, -1)
        return torch.cat([cls_tokens, register_tokens, x], dim=1)

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

    def normalize_patch_embeddings(self, x: Tensor) -> Tensor:
        return self.patch_embed_norm(x) if self.patch_embed_norm is not None else x

    def _validate_conditioning(self, conditioning: Tensor | None) -> None:
        if self.config.conditioning_size is None:
            if conditioning is not None:
                raise ValueError("conditioning is not supported unless config.conditioning_size is set")
        elif conditioning is None:
            raise ValueError("conditioning is required when config.conditioning_size is set")

    def _validate_packed_execution(self) -> None:
        if torch.compiler.is_exporting():
            raise RuntimeError("packed ViT execution does not support torch.export")
        if _EXPLAINABILITY_TRACE_ACTIVE.get():
            raise RuntimeError("packed ViT execution does not support explainability tracing")
        if self.config.token_specialization_enabled:
            raise RuntimeError("packed ViT execution does not support token specialization")
        if self.config.conditioning_size is not None:
            raise RuntimeError("packed ViT execution does not support conditioned MLPs")
        if self._packed_quantization_enabled:
            raise RuntimeError("packed ViT execution does not support quantization")
        if len(self.config.patch_size) != 2:
            raise RuntimeError("packed ViT execution supports 2D inputs only")

    def encode_packed(
        self,
        visual_features: PackedSequence,
        *,
        rope: Tensor | None = None,
        tokenized_size: Sequence[int] | None = None,
        output_norm: bool = True,
        backend: PackedAttentionBackend = "auto",
    ) -> PackedViTFeatures:
        """Encode already tokenized visual features without padding.

        The method applies the configured patch-embedding normalization. When
        RoPE is configured, ``rope`` must contain sine/cosine rows aligned with
        ``visual_features.values``.
        """
        normalized = visual_features.with_values(self.normalize_patch_embeddings(visual_features.values))
        return self._encode_packed_normalized(
            normalized,
            rope=rope,
            tokenized_size=tokenized_size,
            output_norm=output_norm,
            backend=backend,
        )

    def _encode_packed_normalized(
        self,
        visual_features: PackedSequence,
        *,
        rope: Tensor | None,
        tokenized_size: Sequence[int] | None,
        output_norm: bool,
        backend: PackedAttentionBackend,
    ) -> PackedViTFeatures:
        self._validate_packed_execution()
        if self.rope is not None and rope is None:
            raise ValueError("packed RoPE values are required when the ViT uses RoPE")
        if self.rope is None and rope is not None:
            raise ValueError("packed RoPE values were provided to a ViT without RoPE")

        packed, aligned_rope = _add_packed_prefix(
            visual_features,
            self.cls_tokens,
            self.register_tokens,
            rope,
        )
        for block in self.blocks:
            assert isinstance(block, TransformerEncoderLayer)
            if self.config.activation_checkpointing and self.training:
                layout = packed

                def checkpointed_block(
                    values: Tensor,
                    offsets: Tensor,
                    block_rope: Tensor | None,
                    current_block: TransformerEncoderLayer = block,
                    current_layout: PackedSequence = layout,
                ) -> Tensor:
                    block_input = PackedSequence._from_validated(
                        values,
                        offsets,
                        current_layout.lengths,
                        current_layout.min_seqlen,
                        current_layout.max_seqlen,
                    )
                    return current_block.forward_packed(block_input, rope=block_rope, backend=backend).values

                values = cast(
                    Tensor,
                    checkpoint(
                        checkpointed_block,
                        packed.values,
                        packed._cu_seqlens_for_ops(),
                        aligned_rope,
                        use_reentrant=False,
                    ),
                )
                packed = packed.with_values(values)
            else:
                packed = block.forward_packed(packed, rope=aligned_rope, backend=backend)

        values = self.output_norm(packed.values) if output_norm else packed.values
        return _split_packed_features(
            packed.with_values(values),
            self.config.num_cls_tokens,
            self.config.num_register_tokens,
            tokenized_size,
        )

    def forward_packed(
        self,
        images: Tensor,
        mask: Tensor,
        rope_seed: int | None = None,
        output_norm: bool = True,
        *,
        backend: PackedAttentionBackend = "auto",
    ) -> PackedViTFeatures:
        """Encode same-size images using a ragged patch-validity mask."""
        self._validate_packed_execution()
        if images.ndim != 4:
            raise ValueError("forward_packed expects same-size dense 2D images with shape [B, C, H, W]")
        if mask.dtype != torch.bool or mask.device != images.device:
            raise ValueError("packed image mask must be boolean and on the image device")
        tokenized_size = self.stem.tokenized_size(images.shape[2:])
        visual_features = self.stem(images)
        visual_features = self.normalize_patch_embeddings(visual_features)
        if mask.shape != visual_features.shape[:2]:
            raise ValueError(
                f"packed image mask must have shape {tuple(visual_features.shape[:2])}, got {tuple(mask.shape)}"
            )
        packed_visual = PackedSequence.from_padded(visual_features, mask)

        packed_rope = None
        if self.rope is not None:
            dense_rope = self.prepare_rope(tokenized_size, rope_seed=rope_seed)
            batch_size = images.shape[0]
            packed_rope = torch.stack(
                (
                    dense_rope[0].expand(batch_size, -1, -1)[mask],
                    dense_rope[1].expand(batch_size, -1, -1)[mask],
                )
            )
        return self._encode_packed_normalized(
            packed_visual,
            rope=packed_rope,
            tokenized_size=tokenized_size,
            output_norm=output_norm,
            backend=backend,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        rope_seed: int | None = None,
        output_norm: bool = True,
        conditioning: Tensor | None = None,
    ) -> ViTFeatures:
        self._validate_conditioning(conditioning)

        # Prepare transformer input
        tokenized_size = self.stem.tokenized_size(x.shape[2:])
        x = self.stem(x)
        x = self.normalize_patch_embeddings(x)
        x = apply_mask(mask, x) if mask is not None else x
        x = self.add_prefix_tokens(x)

        # Prepare RoPE sin/cos if needed
        rope = self.prepare_rope(tokenized_size, mask, rope_seed) if self.rope is not None else None

        # Apply transformer
        for block in self.blocks:
            assert isinstance(block, TransformerEncoderLayer)
            if self.config.activation_checkpointing and self.training:
                x = cast(Tensor, checkpoint(block, x, rope, conditioning, use_reentrant=False))
            else:
                x = block(x, rope=rope, conditioning=conditioning)

        # Prepare output
        x = self.output_norm(x) if output_norm else x
        return ViTFeatures(x, self.config.num_register_tokens, self.config.num_cls_tokens, tokenized_size)

    if TYPE_CHECKING:

        def __call__(
            self,
            x: Tensor,
            mask: Tensor | None = None,
            rope_seed: int | None = None,
            output_norm: bool = True,
            conditioning: Tensor | None = None,
        ) -> ViTFeatures:
            return self.forward(x, mask, rope_seed, output_norm, conditioning)

    @torch.no_grad()
    def _reshape_attention_weights(
        self, w: Tensor, tokenized_size: Sequence[int], mask: Tensor | None = None
    ) -> Tensor:
        B, H, Lq, Lk = w.shape
        assert Lq == Lk, f"Query and key lengths must match, got {Lq} and {Lk}"
        w = w[..., self.prefix_length :]
        if mask is not None:
            full = w.new_zeros((B, H, Lq, math.prod(tokenized_size)))
            for batch_index in range(B):
                key_count = int(mask[batch_index].sum().item())
                full[batch_index, ..., mask[batch_index]] = w[batch_index, ..., :key_count]
            w = full
        w = w.view(B, H, Lq, *tokenized_size)
        return w.contiguous()

    def forward_attention_weights(
        self,
        x: Tensor,
        conditioning: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        rope_seed: int | None = None,
    ) -> dict[str, Tensor]:
        self._validate_conditioning(conditioning)

        # Prepare transformer input
        tokenized_size = self.stem.tokenized_size(x.shape[2:])
        x = self.stem(x)
        x = self.normalize_patch_embeddings(x)
        x = apply_mask(mask, x) if mask is not None else x
        x = self.add_prefix_tokens(x)
        rope = self.prepare_rope(tokenized_size, mask, rope_seed) if self.rope is not None else None

        # Apply transformer
        weights: dict[str, Tensor] = {}
        for i, block in enumerate(self.blocks):
            assert isinstance(block, TransformerEncoderLayer)
            w_i = block.self_attention.forward_weights(x, rope=rope)
            weights[f"layer_{i}"] = self._reshape_attention_weights(w_i, tokenized_size, mask)
            x = block(x, rope=rope, conditioning=conditioning)

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
        if self.patch_embed_norm is not None:
            self.patch_embed_norm.requires_grad_(requires_grad)
        self.blocks.requires_grad_(requires_grad)
        self.output_norm.requires_grad_(requires_grad)
        if self.register_tokens is not None:
            self.register_tokens.requires_grad_(requires_grad)


register_constructors()
