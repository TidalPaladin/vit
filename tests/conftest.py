import os

import pytest
import torch
import torch._dynamo
import torch._dynamo.config


# Disable torch.compile by default for faster tests
# The env var signals our intent, the config flag actually disables compilation
if os.environ.get("TORCHDYNAMO_DISABLE", "1") == "1":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    torch._dynamo.config.disable = True

torch._dynamo.config.dynamic_shapes = True
torch._dynamo.config.cache_size_limit = 100000000
torch.backends.cuda.matmul.fp32_precision = "high"
torch.backends.cudnn.conv.fp32_precision = "high"  # type: ignore


def pytest_configure(config):
    config.addinivalue_line("markers", "compile: mark test to run with torch.compile enabled")


def cuda_available():
    r"""Checks if CUDA is available and device is ready"""
    if not torch.cuda.is_available():
        return False

    capability = torch.cuda.get_device_capability()
    arch_list = torch.cuda.get_arch_list()
    if isinstance(capability, tuple):
        capability = f"sm_{''.join(str(x) for x in capability)}"

    if capability not in arch_list:
        return False

    return True


def handle_cuda_mark(item):  # pragma: no cover
    has_cuda_mark = any(item.iter_markers(name="cuda"))
    if has_cuda_mark and not cuda_available():
        import pytest

        pytest.skip("Test requires CUDA and device is not ready")


def pytest_runtest_setup(item):
    handle_cuda_mark(item)


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param == "cuda:0" and not cuda_available():
        pytest.skip("Test requires CUDA and device is not ready")
    return torch.device(request.param)


@pytest.fixture(autouse=True)
def handle_compile_marker(request):
    """Enable torch.compile for tests marked with @pytest.mark.compile"""
    if request.node.get_closest_marker("compile"):
        # Temporarily enable dynamo for this test
        os.environ.pop("TORCHDYNAMO_DISABLE", None)
        torch._dynamo.config.disable = False
        torch._dynamo.reset()
        yield
        # Re-disable after test
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        torch._dynamo.config.disable = True
        torch._dynamo.reset()
    else:
        yield
