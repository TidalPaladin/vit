from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Literal, Self, Sequence, Tuple, Type, cast

import torch
import torch.nn as nn
import yaml
from einops.layers.torch import Reduce
from torch import Tensor

from .fused import LayerNormLinear, LayerNormMLP
from .helpers import DEFAULT_BACKEND, Backend, check_te_installed, try_import_te
from .patch_embed import ConvNextPatchEmbed2d, PatchEmbed2d
from .tokens import apply_mask, create_mask
from .transformer import TransformerLayer


if TYPE_CHECKING:
    import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
else:
    te = try_import_te()


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

    # Transformer
    depth: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_gqa_groups: int | None = None
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    normalization: str = "RMSNorm"
    bias: bool = True
    activation: str = "srelu"
    drop_path_rate: float = 0.0
    decoder: bool = False
    decoder_layers: Sequence[int] | None = None
    num_register_tokens: int = 0

    # Other
    checkpoint: bool = False
    backend: Backend = DEFAULT_BACKEND

    # Trainable blocks
    mlp_requires_grad: bool = True
    self_attention_requires_grad: bool = True
    inter_attention_requires_grad: bool = True

    # ConvNext patch embedding
    convnext_patch_embed: bool = False
    convnext_depth: int = 2
    convnext_patch_size: Sequence[int] = field(default_factory=lambda: [2, 2])

    @property
    def device_type(self) -> Literal["cpu", "cuda"]:
        return "cuda" if self.backend == "te" else "cpu"

    @property
    def isotropic_output_dim(self) -> int:
        return self.hidden_size

    def instantiate(self) -> "ViT":
        if self.backend not in ("te", "pytorch"):
            raise ValueError(f"Invalid backend: {self.backend}")
        return ViT(self)

    @property
    def transformer_kwargs(self) -> Dict[str, Any]:
        return dict(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_gqa_groups=self.num_gqa_groups,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            normalization=self.normalization,
            bias=self.bias,
            activation=self.activation,
            drop_path_rate=self.drop_path_rate,
            attn_input_format="bshd",
        )

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
    CONFIG_TYPE: ClassVar[Type[ViTConfig]] = ViTConfig

    def __init__(self, config: ViTConfig):
        super().__init__()
        self._config = config

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(config.hidden_size))
        self.register_tokens = nn.Parameter(torch.randn(config.num_register_tokens, config.hidden_size))

        # Stem tokenizer
        if config.convnext_patch_embed:
            self.stem = ConvNextPatchEmbed2d(
                config.in_channels,
                config.hidden_size,
                cast(Tuple[int, int], tuple(config.patch_size)),
                normalization=config.normalization,
                backend=config.backend,
                depth=config.convnext_depth,
                convnext_patch_size=config.convnext_patch_size,
            )
        else:
            self.stem = PatchEmbed2d(
                config.in_channels,
                config.hidden_size,
                cast(Tuple[int, int], tuple(config.patch_size)),
                normalization=config.normalization,
                backend=config.backend,
            )

        # Transformer blocks
        if self.config.decoder:
            decoder_layers = self.config.decoder_layers or range(config.depth)
        else:
            decoder_layers = []
        decoder_layers = set(decoder_layers)
        self.blocks = nn.ModuleList(
            [
                self.create_decoder_layer(i) if i in decoder_layers else self.create_encoder_layer(i)
                for i in range(config.depth)
            ]
        )

        self.mlp_requires_grad_(self.config.mlp_requires_grad)
        self.self_attention_requires_grad_(self.config.self_attention_requires_grad)
        self.inter_attention_requires_grad_(self.config.inter_attention_requires_grad)

    @property
    def config(self) -> ViTConfig:
        return self._config

    def create_encoder_layer(self, i: int = 0, **kwargs) -> TransformerLayer:
        """
        Creates a Transformer encoder layer.

        This method initializes a Transformer encoder layer with the specified
        parameters. It supports various configurations such as the number of
        attention heads, feedforward dimension, dropout rate, activation functions,
        and more.

        Args:
            i: Index of the encoder layer. Default is 0.

        Keyword Args:
            Additional keyword arguments to override default layer parameters.
        """
        _kwargs = self.config.transformer_kwargs
        _kwargs.update(kwargs)
        _kwargs["layer_number"] = i + 1
        _kwargs["layer_type"] = "encoder"
        match self.config.backend:
            case "pytorch":
                layer = TransformerLayer(**_kwargs)
            case "te":
                check_te_installed(te)
                _kwargs["self_attn_mask_type"] = "no_mask"
                layer = te.TransformerLayer(**_kwargs)
            case _:
                raise ValueError(f"Invalid backend: {self.config.backend}")
        return layer

    def create_decoder_layer(self, i: int = 0, **kwargs) -> TransformerLayer:
        """
        Creates a Transformer decoder layer.

        This method initializes a Transformer decoder layer with the specified
        parameters. It supports various configurations such as the number of
        attention heads, feedforward dimension, dropout rate, activation functions,
        and more.

        Args:
            i: Index of the encoder layer. Default is 0.

        Keyword Args:
            Additional keyword arguments to override default layer parameters.
        """
        _kwargs = self.config.transformer_kwargs
        _kwargs.update(kwargs)
        _kwargs["layer_number"] = i + 1
        _kwargs["layer_type"] = "decoder"
        match self.config.backend:
            case "pytorch":
                layer = TransformerLayer(**_kwargs)
            case "te":
                check_te_installed(te)
                _kwargs["self_attn_mask_type"] = "no_mask"
                layer = te.TransformerLayer(**_kwargs)
            case _:
                raise ValueError(f"Invalid backend: {self.config.backend}")
        return layer

    def create_norm(self, hidden_size: int) -> nn.Module:
        match (self.config.normalization, self.config.backend):
            case ("LayerNorm", "pytorch"):
                return nn.LayerNorm(hidden_size)
            case ("RMSNorm", "pytorch"):
                return nn.RMSNorm(hidden_size)
            case ("LayerNorm", "te"):
                check_te_installed(te)
                return te.LayerNorm(hidden_size)
            case ("RMSNorm", "te"):
                check_te_installed(te)
                return te.RMSNorm(hidden_size)
            case _:
                raise ValueError(f"Invalid normalization: {self.config.normalization}")

    def create_head(
        self,
        out_dim: int,
        pool_type: str | None = None,
        mlp: bool = False,
        activation: str | None = None,
    ) -> nn.Module:
        r"""Creates a head for the model.

        Args:
            out_dim: Dimension of the output.
            pool_type: Type of pooling to apply, or ``None`` to skip pooling.
            mlp: Whether to use a MLP instead of a linear layer.
            activation: Activation function to use (for MLP), or ``None`` to use the backbone's activation.
        """
        layer = nn.Sequential()

        # Pooling type
        match pool_type:
            case "avg":
                layer.add_module("pool", Reduce("b l d -> b d", "mean"))
            case "max":
                layer.add_module("pool", Reduce("b l d -> b d", "max"))
            case None:
                pass
            case _:
                raise ValueError(f"Invalid pool type: {pool_type}")

        # Normalization + Linear
        if mlp:
            layer.add_module("layernorm_mlp", self.create_mlp(out_dim, activation))
        else:
            match self.config.backend:
                case "pytorch":
                    layer.add_module(
                        "layernorm_linear",
                        LayerNormLinear(
                            self.config.isotropic_output_dim,
                            out_dim,
                            bias=self.config.bias,
                            normalization=self.config.normalization,
                        ),
                    )
                case "te":
                    check_te_installed(te)
                    layer.add_module(
                        "layernorm_linear",
                        te.LayerNormLinear(
                            self.config.isotropic_output_dim,
                            out_dim,
                            bias=self.config.bias,
                            normalization=self.config.normalization,
                        ),
                    )
                case _:
                    raise ValueError(f"Invalid backend: {self.config.backend}")

        return layer

    def create_mlp(self, out_dim: int, activation: str | None = None) -> nn.Module:
        r"""Creates a MLP with a final output projection to `out_dim`.

        Args:
            out_dim: Dimension of the output.
            activation: Activation function to use, or ``None`` to use the backbone's activation.

        """
        layer = nn.Sequential()
        match self.config.backend:
            case "pytorch":
                layer.add_module(
                    "mlp",
                    LayerNormMLP(
                        self.config.isotropic_output_dim,
                        self.config.ffn_hidden_size,
                        bias=self.config.bias,
                        normalization=self.config.normalization,
                        activation=activation or self.config.activation,
                    ),
                )
                layer.add_module("dropout", nn.Dropout(self.config.hidden_dropout))
                layer.add_module("output", nn.Linear(self.config.isotropic_output_dim, out_dim))
            case "te":
                check_te_installed(te)
                layer.add_module(
                    "mlp",
                    te.LayerNormMLP(
                        self.config.isotropic_output_dim,
                        self.config.ffn_hidden_size,
                        out_dim,
                        bias=self.config.bias,
                        normalization=self.config.normalization,
                        activation=activation or self.config.activation,
                    ),
                )
                layer.add_module("dropout", nn.Dropout(self.config.hidden_dropout))
                layer.add_module("output", te.Linear(self.config.isotropic_output_dim, out_dim))
            case _:
                raise ValueError(f"Invalid backend: {self.config.backend}")
        return layer

    def create_mask(
        self,
        input: Tensor,
        unmasked_ratio: float,
        scale: int,
    ) -> Tensor:
        r"""Creates a token mask for the input.

        Args:
            input: Input tensor from which to infer mask properties.
                Should be a raw input prior to tokenization.
            unmasked_ratio: Proportion of tokens to leave unmasked.
            scale: Scale of the mask.

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
            device=device,
        )

        return mask

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        encoder_output: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        B, C, *original_size = x.shape
        self.stem.tokenized_size(cast(Any, tuple(original_size)))

        # Tokenize and apply mask
        x = self.stem(x)
        if mask is not None:
            x = apply_mask(mask, x)

        # Add CLS token and register tokens
        x = torch.cat(
            [
                self.cls_token.view(1, 1, self.config.hidden_size).expand(B, -1, -1),
                self.register_tokens.view(1, -1, self.config.hidden_size).expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        # Transformer blocks and output norm
        for block in self.blocks:
            block = cast(TransformerLayer, block)
            x = block(x, encoder_output=encoder_output, checkpoint_core_attention=self.config.checkpoint)

        # Extract CLS token and features, dropping register tokens
        cls_token = x[:, 0, :].contiguous()
        x = x[:, 1 + self.config.num_register_tokens :, :].contiguous()

        return x, cls_token

    def mlp_requires_grad_(self, requires_grad: bool = True) -> None:
        for block in self.blocks:
            layer = cast(nn.Module, block.layernorm_mlp)
            layer.requires_grad_(requires_grad)

    def self_attention_requires_grad_(self, requires_grad: bool = True) -> None:
        for block in self.blocks:
            layer = cast(nn.Module, block.self_attention)
            layer.requires_grad_(requires_grad)

    def inter_attention_requires_grad_(self, requires_grad: bool = True) -> None:
        for block in self.blocks:
            if hasattr(block, "inter_attention") and block.inter_attention is not None:
                layer = cast(nn.Module, block.inter_attention)
                layer.requires_grad_(requires_grad)


register_constructors()
