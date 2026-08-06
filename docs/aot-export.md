# AOT export

`ViT` supports direct inference export with PyTorch 2.13 `torch.export` and packaging with AOTInductor. Exported models
retain the `ViTFeatures` output type, including CLS/register counts and the tokenized spatial size.

## Support matrix

| Token-specialized attention mode | `torch.export` | AOTInductor inference package |
|----------------------------------|----------------|-------------------------------|
| `auto` | Supported | Supported |
| `dynamic` | Supported | Supported |
| `static` | Supported | Supported |
| `static_max_autotune` | Supported | Supported |

The mode controls `torch.compile` execution in Python. Export inlines the functional attention graph, so the selected
runtime wrapper, static batch allowlist, and GEMM autotuning setting are not embedded in the exported program.
AOTInductor applies its own compiler configuration when it packages that program.

The supported contract is inference with fixed channels and spatial geometry. The batch dimension may be dynamic.
Training AOT export, dynamic image geometry, and the experimental `torch.compile(...).aot_compile()` API are not part
of this contract. AOTInductor remains a beta PyTorch API.

## Export and save a dynamic-batch model

The example uses token specialization, but the same API works when specialization is disabled.

```python
from pathlib import Path

import torch

from vit import ViTConfig


config = ViTConfig(
    in_channels=3,
    patch_size=(16, 16),
    img_size=(224, 224),
    depth=12,
    hidden_size=768,
    ffn_hidden_size=3072,
    num_attention_heads=12,
    num_cls_tokens=1,
    num_register_tokens=7,
    specialize_global_token_norms=True,
    specialize_global_token_qkv_blocks=4,
    token_specialized_attention_compile_mode="auto",
)
model = config.instantiate().eval()
example = torch.randn(2, 3, 224, 224, dtype=config.dtype)
batch = torch.export.Dim("batch", min=1, max=16)

exported = torch.export.export(
    model,
    (example,),
    dynamic_shapes=({0: batch},),
)
torch.export.save(exported, Path("vit-export.pt2"))

loaded = torch.export.load(Path("vit-export.pt2")).module()
features = loaded(torch.randn(4, 3, 224, 224, dtype=config.dtype))
print(features.visual_tokens.shape)
```

The exported program checks the declared batch range. Channels and image dimensions must match the export example.
Choose a realistic maximum because wider dynamic ranges can reduce optimization opportunities.

## Package with AOTInductor

Package the same `ExportedProgram` for inference, then load it without the original Python model instance:

```python
package_path = "vit-aoti.pt2"
torch._inductor.aoti_compile_and_package(
    exported,
    package_path=package_path,
)
compiled_model = torch._inductor.aoti_load_package(package_path)
features = compiled_model(torch.randn(4, 3, 224, 224, dtype=config.dtype))
```

Pass the package path as a string. Build the package on the target platform and validate numerical parity using the
same PyTorch version, device, and dtype intended for deployment. Refer to the official PyTorch documentation for
[`torch.export`](https://docs.pytorch.org/docs/stable/export.html) and
[AOTInductor](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html) for platform and packaging details.
The deployment environment must have `vit` installed and imported so the `ViTFeatures` pytree type is registered.

## Static runtime policies and export

Static allowlists protect ordinary Python execution from unexpected concrete graphs:

```python
config = ViTConfig(
    # ... model and specialization parameters ...
    token_specialized_attention_compile_mode="static",
    token_specialized_attention_static_batch_sizes=(2, 4, 8),
)
```

Calling this Python model with batch 3 raises `ValueError`. Exporting with a declared dynamic batch range produces an
independent exported program, so its accepted batches come from the `torch.export.Dim` constraint instead. Document
and validate the runtime contract for the artifact rather than assuming it inherits the Python compile policy.
