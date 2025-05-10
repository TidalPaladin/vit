from typing import TYPE_CHECKING, Sequence, cast

import torch
import torch.nn as nn
from torch import Tensor

from .fused import LayerNormMLP
from .helpers import DEFAULT_BACKEND, Backend, check_te_installed, compile_is_disabled, try_import_te


if TYPE_CHECKING:
    import transformer_engine.pytorch as te  # type: ignore[reportMissingImports]
else:
    te = try_import_te()


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def create_grid(
    dims: Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    normalize: bool = True,
) -> Tensor:
    r"""Create a grid of coordinate values given the size of each dimension.

    Args:
        dims:
            The length of each dimension
        proto:
            If provided, a source tensor with which to match device / requires_grad
        normalize:
            If true, normalize coordinate values on the range :math:`\[-1, 1\]`

    Shapes:
        * Output - :math:`(1, L, C)` where :math:`C` is ``len(dims)`` and :math:`L` is ``product(dims)``
    """
    if normalize:
        lens = [torch.linspace(-1, 1, d, device=device, dtype=dtype) for d in dims]
    else:
        lens = [torch.arange(d, device=device, dtype=dtype) for d in dims]
    grid = torch.stack(torch.meshgrid(lens, indexing="ij"), dim=-1)
    return grid.view(1, -1, len(dims))


class RelativeFactorizedPosition(nn.Module):
    """
    Computes relative factorized position encodings.

    A grid of positions in the interval :math:`[-1, 1]` is first created.
    This grid is then projected into a higher-dimensional space using a single linear projection.

    Args:
        d_in:
            Input dimension size
        d_out:
            Output dimension size

    Shapes:
        * Input - :math:`(C,)` where :math:`C` is the number of input dimensions
        * Output - :math:`(1, L, D)` where :math:`L` is the product of input dimensions and :math:`D` is the output dimension
    """

    def __init__(
        self,
        d_in: int,
        hidden_size: int,
        ffn_hidden_size: int,
        activation: str = "gelu",
        normalization: str = "LayerNorm",
        bias: bool = True,
        backend: Backend = DEFAULT_BACKEND,
    ):
        super().__init__()
        match backend:
            case "pytorch":
                self.linear = nn.Linear(d_in, hidden_size, bias=bias)
                self.mlp = LayerNormMLP(
                    hidden_size, ffn_hidden_size, activation=activation, normalization=normalization, bias=bias
                )
            case "te":
                check_te_installed(te)
                self.linear = te.Linear(d_in, hidden_size, bias=bias)
                self.mlp = te.LayerNormMLP(
                    hidden_size, ffn_hidden_size, activation=activation, normalization=normalization, bias=bias
                )
            case _:
                raise ValueError(f"Backend {backend} not supported")

    def forward(self, dims: Sequence[int]) -> Tensor:
        with torch.no_grad():
            grid = create_grid(dims, device=cast(Tensor, self.linear.weight).device)
        y = self.linear(grid)
        y = self.mlp(y)
        return y


# Taken from TransformerEngine


class RotaryPositionEmbedding(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: int | None = None,
        pretrained_max_position_embeddings: int | None = None,
        rotary_base: float = 10000.0,
    ):
        super().__init__()
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.rotary_base = rotary_base
        inv_freq = 1.0 / (self.rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.pretrained_max_position_embeddings = pretrained_max_position_embeddings

    def forward(self, max_seq_len: int, offset: int = 0):
        """
        Create rotary position embedding frequencies

        Parameters
        ----------
        max_seq_len: int
            sequence length of a sample
        offset: int, default = 0
            fixed offset for freqencies
        """
        seq = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype) + offset

        if self.pretrained_max_position_embeddings is not None and self.seq_len_interpolation_factor is not None:
            if max_seq_len > self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor:
                # dynamic linear scaling (length > position we have learned)
                seq *= 1 / (max_seq_len / self.pretrained_max_position_embeddings)
            else:
                # fixed linear scaling
                seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.einsum("i , j -> i j", seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return emb.reshape(emb.size(0), 1, 1, emb.size(1))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


# Taken from TransformerEngine
def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor, tensor_format: str = "sbhd") -> torch.Tensor:
    assert tensor_format in ("sbhd", "bshd"), (
        "Only formats `sbhd` or `bshd` are supported for input tensor `t` " f"when fused is False, got {tensor_format}."
    )

    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert cur_seq_len <= max_seq_len, f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    freqs = freqs[:cur_seq_len]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
    # cos/sin first then dtype conversion for better precision
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)
