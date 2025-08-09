from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import _permute_and_fold_head, _unfold_head_and_permute
from .helpers import get_activation


def batched_linear(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    y = torch.einsum("...sio,...si->...so", weight, x)
    if bias is not None:
        y = y + bias
    return y


def dispatch(tokens: Tensor, slots: Tensor, scale: Tensor) -> Tensor:
    # FIXME: placeholder for proper head_dim selection
    if tokens.shape[-1] % 64 == 0:
        head_dim = 64
    elif tokens.shape[-1] % 32 == 0:
        head_dim = 32
    else:
        head_dim = 16
    norm_tokens = F.normalize(tokens, dim=-1, eps=1e-5)
    slots = F.normalize(slots, dim=-1, eps=1e-5) * scale.unsqueeze(-1)
    tokens = _unfold_head_and_permute(tokens, head_dim=head_dim)
    norm_tokens = _unfold_head_and_permute(norm_tokens, head_dim=head_dim)
    slots = _unfold_head_and_permute(slots.unsqueeze(0), head_dim=head_dim)
    slots = F.scaled_dot_product_attention(slots, norm_tokens, tokens, scale=1.0)
    return _permute_and_fold_head(slots)


def combine(moe_output: Tensor, slots: Tensor, tokens: Tensor, scale: Tensor) -> Tensor:
    # FIXME: placeholder for proper head_dim selection
    if tokens.shape[-1] % 64 == 0:
        head_dim = 64
    elif tokens.shape[-1] % 32 == 0:
        head_dim = 32
    else:
        head_dim = 16
    norm_tokens = F.normalize(tokens, dim=-1, eps=1e-5)
    slots = F.normalize(slots, dim=-1, eps=1e-5) * scale.unsqueeze(-1)
    norm_tokens = _unfold_head_and_permute(norm_tokens, head_dim=head_dim)
    slots = _unfold_head_and_permute(slots.unsqueeze(0), head_dim=head_dim)
    moe_output = _unfold_head_and_permute(moe_output, head_dim=head_dim)
    moe_output = F.scaled_dot_product_attention(norm_tokens, slots, moe_output, scale=1.0)
    return _permute_and_fold_head(moe_output)


@torch.compile(
    fullgraph=True,
    dynamic=False,
    options={
        "layout_optimization": True,
        "epilogue_fusion": True,
        "aggressive_fusion": True,
    },
)
def norm_mlp_softmoe(
    # fmt: off
    x: Tensor, slots: Tensor, scale: Tensor,
    fc1_weight: Tensor, fc1_bias: Tensor | None,
    fc2_weight: Tensor, fc2_bias: Tensor | None,
    norm_weight: Tensor | None,
    activation: Callable[[Tensor], Tensor],
    eps: float,
    dropout: float,
    training: bool,
    # fmt: on
) -> Tensor:
    if norm_weight is not None:
        x = F.rms_norm(x, x.shape[-1:], weight=norm_weight, eps=eps)

    y = dispatch(x, slots, scale)
    y = batched_linear(y, fc1_weight, fc1_bias)
    y = activation(y)
    y = F.dropout(y, p=dropout, training=training)
    y = batched_linear(y, fc2_weight, fc2_bias)
    y = F.dropout(y, p=dropout, training=training, inplace=True)
    y = combine(y, slots, x, scale)
    return y


@torch.compile(
    fullgraph=True,
    dynamic=False,
    options={
        "layout_optimization": True,
        "epilogue_fusion": True,
        "aggressive_fusion": True,
    },
)
def norm_mlp_softmoe_glu(
    # fmt: off
    x: Tensor, slots: Tensor, scale: Tensor,
    fc1_weight: Tensor, fc1_bias: Tensor | None,
    fc2_weight: Tensor, fc2_bias: Tensor | None,
    norm_weight: Tensor | None,
    activation: Callable[[Tensor], Tensor],
    eps: float,
    dropout: float,
    training: bool,
    limit: float | None = None,
    extra_bias: float | None = None,
    # fmt: on
) -> Tensor:
    if norm_weight is not None:
        x = F.rms_norm(x, x.shape[-1:], weight=norm_weight, eps=eps)

    y = dispatch(x, slots, scale)

    # FC1 - GLU
    y = batched_linear(y, fc1_weight, fc1_bias)
    y_linear, y_glu = y.chunk(2, dim=-1)
    if limit is not None:
        y_linear = y_linear.clamp(min=-limit, max=limit)
        y_glu = y_glu.clamp(min=None, max=limit)
    if extra_bias is not None:
        y_linear = y_linear + extra_bias
    y = activation(y_glu) * y_linear
    y = F.dropout(y, p=dropout, training=training)

    # FC2
    y = batched_linear(y, fc2_weight, fc2_bias)
    y = F.dropout(y, p=dropout, training=training, inplace=True)

    y = combine(y, slots, x, scale)
    return y


class SoftMoE(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_slots: int,
        bias: bool = True,
        activation: str = "gelu",
        eps: float = 1e-5,
        dropout: float = 0.1,
        limit: float | None = None,
        extra_bias: float | None = None,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self._is_glu = activation.endswith("glu")
        self.fc1_weight = nn.Parameter(
            torch.randn(num_slots, hidden_size, ffn_hidden_size if not self._is_glu else 2 * ffn_hidden_size)
        )
        self.fc1_bias = (
            nn.Parameter(torch.zeros(num_slots, ffn_hidden_size if not self._is_glu else 2 * ffn_hidden_size))
            if bias
            else None
        )
        self.fc2_weight = nn.Parameter(torch.randn(num_slots, ffn_hidden_size, hidden_size))
        self.fc2_bias = nn.Parameter(torch.zeros(num_slots, hidden_size)) if bias else None
        self.slots = nn.Parameter(torch.empty(num_slots, hidden_size))
        self.scale = nn.Parameter(torch.empty(num_slots))

        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        self.limit = limit
        self.extra_bias = extra_bias
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        nn.init.normal_(self.slots, std=0.02)
        nn.init.ones_(self.scale)
        nn.init.trunc_normal_(self.fc1_weight, std=0.02)
        nn.init.trunc_normal_(self.fc2_weight, std=0.02)
        if self.fc1_bias is not None:
            nn.init.zeros_(self.fc1_bias)
        if self.fc2_bias is not None:
            nn.init.zeros_(self.fc2_bias)

    def forward(self, x: Tensor) -> Tensor:
        if self._is_glu:
            return norm_mlp_softmoe_glu(
                # fmt: off
                x, self.slots, self.scale,
                self.fc1_weight, self.fc1_bias,
                self.fc2_weight, self.fc2_bias,
                self.norm.weight,
                self.activation,
                self.norm.eps or 1e-5,
                self.dropout.p,
                self.training,
                self.limit,
                self.extra_bias,
                # fmt: on
            )
        else:
            return norm_mlp_softmoe(
                # fmt: off
                x, self.slots, self.scale,
                self.fc1_weight, self.fc1_bias,
                self.fc2_weight, self.fc2_bias,
                self.norm.weight,
                self.activation,
                self.norm.eps or 1e-5,
                self.dropout.p,
                self.training,
                # fmt: on
            )
