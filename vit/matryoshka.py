from dataclasses import dataclass

from torch import Tensor


def slice_matryoshka(x: Tensor, frac: float | None = None) -> Tensor:
    if frac is None or frac == 1.0:
        return x
    D = x.shape[-1]
    D_sliced = int(D * frac)
    return x[..., :D_sliced]


def slice_matryoshka_weight(w: Tensor, input_frac: float | None = None, output_frac: float | None = None) -> Tensor:
    if input_frac is None and output_frac is None or input_frac == 1.0 and output_frac == 1.0:
        return w
    else:
        D_out, D_in = w.shape
        D_in_sliced = int(D_in * input_frac) if input_frac is not None else D_in
        D_out_sliced = int(D_out * output_frac) if output_frac is not None else D_out
        return w[..., :D_out_sliced, :D_in_sliced]


def slice_matryoshka_heads(x: Tensor, frac: float | None = None) -> Tensor:
    if frac is None or frac == 1.0:
        return x
    else:
        # b h l d
        H = x.shape[1]
        H_sliced = int(H * frac)
        return x[..., :H_sliced, :, :]


def add_sliced_features(x1: Tensor, x2: Tensor) -> Tensor:
    D1, D2 = x1.shape[-1], x2.shape[-1]
    if D1 == D2:
        return x1 + x2
    else:
        sliced = x1 if D1 < D2 else x2
        unsliced = x2 if D1 < D2 else x1
        unsliced[..., :D1] = unsliced[..., :D1] + sliced
        return unsliced


@dataclass
class MatryoshkaConfig:
    feature_frac: float | None = None
    feedforward_frac: float | None = None
    heads_frac: float | None = None

    def __post_init__(self):
        if self.feature_frac is not None and not 0 < self.feature_frac <= 1:
            raise ValueError("feature_frac must be between 0 and 1")
        if self.feedforward_frac is not None and not 0 < self.feedforward_frac <= 1:
            raise ValueError("feedforward_frac must be between 0 and 1")
        if self.heads_frac is not None and not 0 < self.heads_frac <= 1:
            raise ValueError("heads_frac must be between 0 and 1")

    def __call__(self, x: Tensor) -> Tensor:
        if self.feature_frac is not None:
            x = slice_matryoshka(x, self.feature_frac)
        return x
