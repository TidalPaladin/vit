from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchao.quantization import FqnToConfig, quantize_
from torchao.quantization.qat import FakeQuantizedLinear


def quantize_linears(owner: nn.Module, names: Sequence[str], config: Any) -> None:
    """Apply module replacements through their parent so TorchAO retains them."""
    quantize_(owner, FqnToConfig(OrderedDict.fromkeys(names, config)), filter_fn=None)


def qat_linear(module: nn.Linear | None) -> nn.Linear | None:
    return module if isinstance(module, FakeQuantizedLinear) else None


def linear_forward(x: Tensor, weight: Tensor, bias: Tensor | None, module: nn.Linear | None = None) -> Tensor:
    """Call QAT modules so their activation and weight fake quantizers execute."""
    return F.linear(x, weight, bias) if module is None else module(x)
