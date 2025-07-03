import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit.fused import NormMLP, norm_mlp
from vit.matryoshka import slice_matryoshka, slice_matryoshka_weight, unslice_matryoshka


class TestMatryoshka:

    @pytest.mark.parametrize("frac", [1.0, 0.5, 0.25])
    def test_forward_linear(self, device, frac):
        D_hidden, D_feedforward = 10, 20
        layer = nn.Linear(D_hidden, D_feedforward).to(device)

        B, L = 2, 10
        x = torch.randn(B, L, D_hidden, device=device)

        # Slicing
        x_sliced = slice_matryoshka(x, frac)
        w_sliced = slice_matryoshka_weight(layer.weight, frac, frac)
        b_sliced = slice_matryoshka(layer.bias, frac)
        assert x_sliced.shape[-1] == int(D_hidden * frac)
        assert w_sliced.shape[-2] == int(D_feedforward * frac)
        assert w_sliced.shape[-1] == int(D_hidden * frac)
        assert b_sliced.shape[-1] == int(D_feedforward * frac)

        # Forward
        y_sliced = F.linear(x_sliced, w_sliced, b_sliced)
        assert y_sliced.shape == (B, L, int(D_feedforward * frac))

    @pytest.mark.parametrize("frac", [1.0, 0.5, 0.25])
    def test_backward_linear(self, device, frac):
        D_hidden, D_feedforward = 10, 20
        layer = nn.Linear(D_hidden, D_feedforward).to(device)

        B, L = 2, 10
        x = torch.randn(B, L, D_hidden, device=device)

        x_sliced = slice_matryoshka(x, frac)
        w_sliced = slice_matryoshka_weight(layer.weight, frac, frac)
        b_sliced = slice_matryoshka(layer.bias, frac)
        y_sliced = F.linear(x_sliced, w_sliced, b_sliced)
        y_sliced.sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

    @pytest.mark.parametrize("frac", [1.0, 0.5, 0.25])
    def test_forward_mlp(self, device, frac):
        D_hidden, D_feedforward = 10, 20
        layer = NormMLP(D_hidden, D_feedforward).to(device)

        B, L = 2, 10
        x = torch.randn(B, L, D_hidden, device=device)

        # Slicing
        x_sliced = slice_matryoshka(x, frac)
        w_norm_sliced = slice_matryoshka(layer.norm.weight, frac)
        w1_sliced = slice_matryoshka_weight(layer.fc1.weight, frac, frac)
        b1_sliced = slice_matryoshka(layer.fc1.bias, frac)
        w2_sliced = slice_matryoshka_weight(layer.fc2.weight, frac, frac)
        b2_sliced = slice_matryoshka(layer.fc2.bias, frac)
        assert x_sliced.shape[-1] == int(D_hidden * frac)
        assert w_norm_sliced.shape[-1] == int(D_hidden * frac)
        assert w1_sliced.shape[-2] == int(D_feedforward * frac)
        assert w1_sliced.shape[-1] == int(D_hidden * frac)
        assert b1_sliced.shape[-1] == int(D_feedforward * frac)
        assert w2_sliced.shape[-1] == int(D_feedforward * frac)
        assert w2_sliced.shape[-2] == int(D_hidden * frac)
        assert b2_sliced.shape[-1] == int(D_hidden * frac)

        # Forward
        y_sliced = norm_mlp(
            x_sliced,
            w1_sliced,
            b1_sliced,
            w2_sliced,
            b2_sliced,
            w_norm_sliced,
            layer.activation,
            layer.norm.eps or 1e-5,
            layer.dropout.p,
            layer.training,
        )
        assert y_sliced.shape == (B, L, int(D_hidden * frac))

        # Residual
        out = unslice_matryoshka(y_sliced, D_hidden)
        assert out.shape == x.shape

    @pytest.mark.parametrize("frac", [1.0, 0.5, 0.25])
    def test_backward_mlp(self, device, frac):
        D_hidden, D_feedforward = 10, 20
        layer = NormMLP(D_hidden, D_feedforward).to(device)

        B, L = 2, 10
        x = torch.randn(B, L, D_hidden, device=device)

        # Slicing
        x_sliced = slice_matryoshka(x, frac)
        w_norm_sliced = slice_matryoshka(layer.norm.weight, frac)
        w1_sliced = slice_matryoshka_weight(layer.fc1.weight, frac, frac)
        b1_sliced = slice_matryoshka(layer.fc1.bias, frac)
        w2_sliced = slice_matryoshka_weight(layer.fc2.weight, frac, frac)
        b2_sliced = slice_matryoshka(layer.fc2.bias, frac)
        y_sliced = norm_mlp(
            x_sliced,
            w1_sliced,
            b1_sliced,
            w2_sliced,
            b2_sliced,
            w_norm_sliced,
            layer.activation,
            layer.norm.eps or 1e-5,
            layer.dropout.p,
            layer.training,
        )
        out = unslice_matryoshka(y_sliced, D_hidden)
        out.sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
