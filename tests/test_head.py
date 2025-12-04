import pytest
import torch

from vit.head import (
    Head,
    HeadConfig,
    TransposedConv2dHead,
    TransposedConv2dHeadConfig,
    UpsampleHead,
    UpsampleHeadConfig,
)
from vit.vit import ViTConfig


class TestHeadConfig:

    def test_instantiate(self):
        config = HeadConfig(out_features=128)
        vit_config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(224, 224),
            depth=3,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=128 // 16,
        )
        model = config.instantiate(vit_config)
        assert isinstance(model, Head)
        assert model.proj.in_features == 128
        assert model.proj.out_features == 128

    def test_instantiate_custom_dims(self):
        config = HeadConfig(in_features=256, out_features=64, dropout=0.1)
        vit_config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(224, 224),
            depth=3,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=128 // 16,
        )
        model = config.instantiate(vit_config)
        assert isinstance(model, Head)
        assert model.proj.in_features == 256
        assert model.proj.out_features == 64


class TestTransposedConv2dHeadConfig:

    def test_instantiate(self):
        config = TransposedConv2dHeadConfig(out_features=64, kernel_size=4, stride=2, padding=1)
        vit_config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(224, 224),
            depth=3,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=128 // 16,
        )
        model = config.instantiate(vit_config)
        assert isinstance(model, TransposedConv2dHead)
        assert model.conv_transpose.in_channels == 128
        assert model.conv_transpose.out_channels == 64

    @pytest.mark.parametrize(
        "kernel_size,stride,padding,output_padding",
        [
            (4, 2, 1, 0),
            (3, 1, 1, 0),
            ((4, 4), (2, 2), (1, 1), (0, 0)),
            (2, 2, 0, 0),
        ],
    )
    def test_instantiate_conv_params(self, kernel_size, stride, padding, output_padding):
        config = TransposedConv2dHeadConfig(
            in_features=64,
            out_features=32,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        vit_config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(224, 224),
            depth=3,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=128 // 16,
        )
        model = config.instantiate(vit_config)
        assert isinstance(model, TransposedConv2dHead)
        assert model.conv_transpose.in_channels == 64
        assert model.conv_transpose.out_channels == 32


class TestHead:

    @pytest.mark.parametrize("in_features", [64, 128])
    @pytest.mark.parametrize("out_features", [32, 128])
    def test_forward(self, device, in_features, out_features):
        x = torch.randn(2, 196, in_features, device=device)
        model = Head(in_features, out_features).to(device)
        out = model(x)
        assert out.shape == (2, 196, out_features)

    def test_forward_2d(self, device):
        x = torch.randn(2, 128, device=device)
        model = Head(128, 64).to(device)
        out = model(x)
        assert out.shape == (2, 64)

    def test_backward(self, device):
        x = torch.randn(2, 196, 128, device=device, requires_grad=True)
        model = Head(128, 64).to(device)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5])
    def test_dropout(self, device, dropout):
        x = torch.randn(2, 196, 128, device=device)
        model = Head(128, 64, dropout=dropout).to(device)
        assert model.dropout.p == dropout
        out = model(x)
        assert out.shape == (2, 196, 64)


class TestTransposedConv2dHead:

    @pytest.mark.parametrize("in_channels", [64, 128])
    @pytest.mark.parametrize("out_channels", [32, 64])
    def test_forward(self, device, in_channels, out_channels):
        x = torch.randn(2, in_channels, 14, 14, device=device)
        model = TransposedConv2dHead(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        ).to(device)
        out = model(x)
        assert out.shape == (2, out_channels, 28, 28)

    @pytest.mark.parametrize(
        "kernel_size,stride,padding,expected_size",
        [
            (4, 2, 1, 28),  # 14 -> 28 (2x upscale)
            (2, 2, 0, 28),  # 14 -> 28 (2x upscale)
            (4, 4, 0, 56),  # 14 -> 56 (4x upscale)
        ],
    )
    def test_output_sizes(self, device, kernel_size, stride, padding, expected_size):
        x = torch.randn(2, 64, 14, 14, device=device)
        model = TransposedConv2dHead(
            in_channels=64,
            out_channels=32,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ).to(device)
        out = model(x)
        assert out.shape == (2, 32, expected_size, expected_size)

    def test_backward(self, device):
        x = torch.randn(2, 64, 14, 14, device=device, requires_grad=True)
        model = TransposedConv2dHead(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
        ).to(device)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.mark.parametrize("groups", [1, 2, 4])
    def test_groups(self, device, groups):
        in_channels = 64
        out_channels = 32
        x = torch.randn(2, in_channels, 14, 14, device=device)
        model = TransposedConv2dHead(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            groups=groups,
        ).to(device)
        out = model(x)
        assert out.shape == (2, out_channels, 28, 28)

    @pytest.mark.parametrize("dilation", [1, 2])
    def test_dilation(self, device, dilation):
        x = torch.randn(2, 64, 14, 14, device=device)
        model = TransposedConv2dHead(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        ).to(device)
        out = model(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 32

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5])
    def test_dropout(self, device, dropout):
        x = torch.randn(2, 64, 14, 14, device=device)
        model = TransposedConv2dHead(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
            dropout=dropout,
        ).to(device)
        assert model.dropout.p == dropout
        out = model(x)
        assert out.shape == (2, 32, 28, 28)


class TestUpsampleHeadConfig:

    def test_instantiate(self):
        config = UpsampleHeadConfig(out_features=32, num_upsample_stages=4)
        vit_config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(224, 224),
            depth=3,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=128 // 16,
        )
        model = config.instantiate(vit_config)
        assert isinstance(model, UpsampleHead)
        assert len(model.upsample_layers) == 4
        assert len(model.smooth_layers) == 4

    @pytest.mark.parametrize("num_upsample_stages", [1, 2, 3, 4])
    def test_instantiate_stages(self, num_upsample_stages):
        config = UpsampleHeadConfig(
            in_features=64,
            out_features=32,
            num_upsample_stages=num_upsample_stages,
        )
        vit_config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(224, 224),
            depth=3,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=128 // 16,
        )
        model = config.instantiate(vit_config)
        assert isinstance(model, UpsampleHead)
        assert len(model.upsample_layers) == num_upsample_stages


class TestUpsampleHead:

    @pytest.mark.parametrize("in_channels", [64, 128])
    @pytest.mark.parametrize("out_channels", [32, 64])
    def test_forward(self, device, in_channels, out_channels):
        # 14x14 input with 4 stages -> 224x224 output (16x upscale)
        x = torch.randn(2, in_channels, 14, 14, device=device)
        model = UpsampleHead(
            in_channels=in_channels,
            out_channels=out_channels,
            num_upsample_stages=4,
        ).to(device)
        out = model(x)
        assert out.shape == (2, out_channels, 224, 224)

    @pytest.mark.parametrize(
        "num_upsample_stages,expected_scale",
        [
            (1, 2),
            (2, 4),
            (3, 8),
            (4, 16),
        ],
    )
    def test_output_sizes(self, device, num_upsample_stages, expected_scale):
        x = torch.randn(2, 64, 14, 14, device=device)
        model = UpsampleHead(
            in_channels=64,
            out_channels=32,
            num_upsample_stages=num_upsample_stages,
        ).to(device)
        out = model(x)
        assert out.shape == (2, 32, 14 * expected_scale, 14 * expected_scale)

    @pytest.mark.parametrize("num_smooth_layers", [1, 2, 3])
    def test_smooth_layers(self, device, num_smooth_layers):
        x = torch.randn(2, 64, 14, 14, device=device)
        model = UpsampleHead(
            in_channels=64,
            out_channels=32,
            num_upsample_stages=2,
            num_smooth_layers=num_smooth_layers,
        ).to(device)
        out = model(x)
        assert out.shape == (2, 32, 56, 56)
        # Check that each stage has the right number of conv layers in smooth
        for smooth in model.smooth_layers:
            conv_count = sum(1 for m in smooth if isinstance(m, torch.nn.Conv2d))  # type: ignore
            assert conv_count == num_smooth_layers

    def test_hidden_channels(self, device):
        x = torch.randn(2, 128, 14, 14, device=device)
        model = UpsampleHead(
            in_channels=128,
            out_channels=32,
            hidden_channels=64,
            num_upsample_stages=3,
        ).to(device)
        out = model(x)
        assert out.shape == (2, 32, 112, 112)
        # Check intermediate channel dimensions
        assert model.upsample_layers[0].in_channels == 128
        assert model.upsample_layers[0].out_channels == 64
        assert model.upsample_layers[1].in_channels == 64
        assert model.upsample_layers[1].out_channels == 64
        assert model.upsample_layers[2].in_channels == 64
        assert model.upsample_layers[2].out_channels == 32

    def test_backward(self, device):
        x = torch.randn(2, 64, 14, 14, device=device, requires_grad=True)
        model = UpsampleHead(
            in_channels=64,
            out_channels=32,
            num_upsample_stages=4,
        ).to(device)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5])
    def test_dropout(self, device, dropout):
        x = torch.randn(2, 64, 14, 14, device=device)
        model = UpsampleHead(
            in_channels=64,
            out_channels=32,
            num_upsample_stages=2,
            dropout=dropout,
        ).to(device)
        assert model.dropout.p == dropout
        out = model(x)
        assert out.shape == (2, 32, 56, 56)
