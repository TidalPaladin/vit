from typing import Any, ClassVar, Tuple, Type, cast

import torch
import torch.nn as nn
import transformer_engine.pytorch as te  # type: ignore
from einops.layers.torch import Reduce
from torch import Tensor

from ..helpers import DEFAULT_TRUNC_STD
from ..tokens import apply_mask, create_mask
from ..vit import ViTConfig
from .patch_embed import PatchEmbed2d


class ViT(nn.Module):
    CONFIG_TYPE: ClassVar[Type[ViTConfig]] = ViTConfig

    def __init__(self, config: ViTConfig):
        super().__init__()
        self._config = config

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(config.hidden_size))

        # Stem tokenizer
        self.stem = PatchEmbed2d(
            config.in_channels,
            config.hidden_size,
            cast(Tuple[int, int], tuple(config.patch_size)),
            normalization=config.normalization,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                self.create_encoder_layer(i) if not self.config.decoder else self.create_decoder_layer(i)
                for i in range(config.depth)
            ]
        )

    @property
    def config(self) -> ViTConfig:
        return self._config

    def create_encoder_layer(self, i: int = 0, **kwargs) -> te.TransformerLayer:
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
        layer = te.TransformerLayer(**_kwargs)
        return layer

    def create_norm(self, hidden_size: int) -> nn.Module:
        match self.config.normalization:
            case "LayerNorm":
                return te.LayerNorm(hidden_size)
            case "RMSNorm":
                return te.RMSNorm(hidden_size)
            case _:
                raise ValueError(f"Invalid normalization: {self.config.normalization}")

    def create_decoder_layer(self, i: int = 0, **kwargs) -> te.TransformerLayer:
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
        layer = te.TransformerLayer(**_kwargs)
        return layer

    def create_head(self, out_dim: int, pool_type: str | None = None) -> nn.Module:
        r"""Creates a head for the model.

        Args:
            out_dim: Dimension of the output.
            pool_type: Type of pooling to apply, or ``None`` to skip pooling.

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
        layer.add_module("norm", self.create_norm(self.config.isotropic_output_dim))
        linear = te.Linear(self.config.isotropic_output_dim, out_dim)
        nn.init.trunc_normal_(linear.weight, std=DEFAULT_TRUNC_STD)
        layer.add_module("linear", linear)

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

        # Add CLS token
        x = torch.cat([self.cls_token.view(1, 1, -1).expand(B, -1, -1), x], dim=1)

        # Transformer blocks and output norm
        for block in self.blocks:
            block = cast(te.TransformerLayer, block)
            x = block(x, encoder_output=encoder_output, checkpoint_core_attention=self.config.checkpoint)

        # Extract CLS token
        cls_token = x[:, 0, :].contiguous()
        x = x[:, 1:, :].contiguous()

        return x, cls_token
