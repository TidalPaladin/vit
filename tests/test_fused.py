import pytest
import torch
from torch.testing import assert_close


class TestLayerNormLinear:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, dtype):
        layer_norm_linear = LayerNormLinear(10, 20)
        x = torch.randn(10)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = layer_norm_linear(x)
        assert y.shape == (20,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, dtype):
        layer_norm_linear = LayerNormLinear(10, 20)
        x = torch.randn(10, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = layer_norm_linear(x)
        y.sum().backward()
        for param in layer_norm_linear.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.cuda
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("parameters_split", [None, {"query": 10, "key": 5, "value": 5}])
    @pytest.mark.parametrize("bias", [False, True])
    def test_baseline(self, normalization, parameters_split, bias):
        if te is None:
            pytest.skip("Transformer Engine is not available")

        layer = LayerNormLinear(10, 20, normalization=normalization, parameters_split=parameters_split, bias=bias).to(
            "cuda"
        )
        baseline = te.LayerNormLinear(
            10, 20, normalization=normalization, parameters_split=parameters_split, bias=bias
        ).to("cuda")

        layer.eval()
        baseline.eval()

        # Sync weights
        for name, param in baseline.named_parameters():
            layer.get_parameter(name).data.copy_(param.data)

        x = torch.randn(8, 10, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            y = layer(x)
            y_baseline = baseline(x)
        assert_close(y, y_baseline)


class TestLayerNormMLP:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("activation", ["relu", "silu", "gelu", "srelu"])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_forward(self, dtype, activation, normalization):
        layer_norm_mlp = LayerNormMLP(10, 20, activation=activation, normalization=normalization)
        x = torch.randn(10)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = layer_norm_mlp(x)
        assert y.shape == (10,)

    def test_determinstic(self):
        torch.random.manual_seed(0)
        layer = LayerNormMLP(10, 20)
        x = torch.randn(10)

        layer.eval()
        y3 = layer(x)
        y4 = layer(x)
        assert_close(y3, y4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, dtype):
        layer_norm_mlp = LayerNormMLP(10, 20)
        x = torch.randn(10, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = layer_norm_mlp(x)
        y.sum().backward()
        for param in layer_norm_mlp.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.cuda
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("bias", [False, True])
    def test_baseline(self, normalization, bias):
        if te is None:
            pytest.skip("Transformer Engine is not available")

        layer = LayerNormMLP(10, 20, normalization=normalization, bias=bias).to("cuda")
        baseline = te.LayerNormMLP(10, 20, normalization=normalization, bias=bias).to("cuda")

        layer.eval()
        baseline.eval()

        # Sync weights
        for name, param in baseline.named_parameters():
            layer.get_parameter(name).data.copy_(param.data)

        x = torch.randn(8, 10, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            y = layer(x)
            y_baseline = baseline(x)
        assert_close(y, y_baseline)


class TestLinear:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, dtype):
        layer = Linear(10, 20)
        x = torch.randn(10)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = layer(x)
        assert y.shape == (20,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward(self, dtype):
        layer = Linear(10, 20)
        x = torch.randn(10, dtype=dtype)
        with torch.autocast(device_type="cpu", dtype=dtype):
            y = layer(x)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
            assert not param.grad.isnan().any()

    @pytest.mark.cuda
    @pytest.mark.parametrize("parameters_split", [None, {"query": 10, "key": 5, "value": 5}])
    @pytest.mark.parametrize("bias", [False, True])
    def test_baseline(self, parameters_split, bias):
        if te is None:
            pytest.skip("Transformer Engine is not available")

        layer = Linear(10, 20, parameters_split=parameters_split, bias=bias).to("cuda")
        baseline = te.Linear(10, 20, parameters_split=parameters_split, bias=bias).to("cuda")

        layer.eval()
        baseline.eval()

        # Sync weights
        for name, param in baseline.named_parameters():
            layer.get_parameter(name).data.copy_(param.data)

        x = torch.randn(8, 10, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            y = layer(x)
            y_baseline = baseline(x)
        assert_close(y, y_baseline)
