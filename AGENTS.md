# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `vit/` (for example `vit/vit.py`, `vit/attention.py`, `vit/transformer.py`).
The Python-first explainability toolbox lives in `vit/explain/`; its sparse autoencoder remains isolated under
`vit/explain/experimental/`.
Unit tests live in `tests/` and follow module-level coverage (`tests/test_vit.py`, `tests/test_attention.py`, etc.).  
Benchmark tooling and CLI entrypoints live in `benchmark/`; generated benchmark outputs are typically written to
`benchmark_results/`.  
Build metadata and tooling configuration are in `pyproject.toml`, `Makefile`, and `.github/workflows/`.

## Architecture & Core Patterns
Main flow is `Images -> PatchEmbed -> Transformer -> ViTFeatures -> Heads`.

- `ViT.forward()` returns a `ViTFeatures` container (not a raw tensor).
- CLS and register tokens are optional. Pooling and classification behavior remains explicit in heads.
- Prefer config-driven construction via `ViTConfig.instantiate()` and `HeadConfig.instantiate()`.
- Use `activation_checkpointing=True` in `ViTConfig` when trading latency for lower training memory.
- Keep packed variable-length execution opt-in. Preserve dense `ViT.forward()` and `ViTFeatures`. Run non-attention
  work on flat values. Construct jagged Q/K/V only at the SDPA boundary. Keep the PyTorch packed helper
  `fullgraph=True, dynamic=True`. Do not share its compiler code object with dense or optional backend helpers.
- Keep packed CLS/register outputs dense and visual tokens as `PackedSequence`. Require explicit `to_padded()` calls.
  Reject token specialization, conditioning, quantization, export, and explainability tracing before kernel launch.
  Also reject 3D inputs, FP32, pre-Ampere CUDA, empty sequences, and malformed offsets.
- Preserve sequence-level stochastic depth, aligned 2D RoPE with identity prefix rotations, training dropout, and
  activation checkpointing on packed values/offsets/RoPE. Use `PackedBatchBudget` and a fingerprinted calibration
  result to reject memory outliers before production steps. Catch OOM only in disposable calibration trials.
- Retain a packed production backend only after three same-GPU decision runs in
  `vit-packed-attention-benchmark`. Require at least 10% lower median latency. Alternatively, require latency within 3%
  and peak memory at least 15% lower. No case can be more than 5% slower than the best baseline. Amortize one packing
  operation across the 12 ViT-S encoder blocks. Measure memory with only the selected method's input representation.
  Isolate bounded baseline compiler helpers for each benchmark case.
- Token specialization is disabled by default and does not change the `ViT.forward()` or `ViTFeatures` contracts.
  When enabled, treat the leading CLS and register tokens as one global stream. Split the pre-attention and pre-MLP
  norms, plus configured LayerScale parameters, in every encoder block. Split QKV only in the configured leading block
  count. Clone each visual branch from its global branch so specialized and shared models are identical at
  initialization. Keep attention, output projections, MLP projections, and the final output norm shared.
  Keep `auto` on the adapting compiled path except for its isolated large-batch or configured-batch training fallback.
  Keep forced dynamic, static, and static-max-autotune wrappers on distinct code objects so their compiler caches do
  not overlap. During `torch.export`, inline the functional attention graph and preserve `ViTFeatures` as a pytree;
  runtime compile modes and static batch allowlists do not become exported-program constraints.

### Explainability architecture

- Stable explainability supports the native repository `ViT` on 2D inputs. Reject 3D inputs with an actionable error.
- Keep normal inference and default traces on the fused MLP path. Capture graph-connected attention probabilities
  and opt-in MLP internals only through the eager trace path in `vit/explain/trace.py`.
- Preserve caller-owned training flags, parameter gradient flags, and existing gradients around explanation calls.
- Route masks, RoPE seeds, output-norm choices, and conditioning through `ForwardArgs` on every explanatory forward.
- Keep raw attribution values unnormalized. Put interpolation and normalization in explicit visualization functions.
- Treat raw attention and attention rollout as query-selected structure, not class attribution or causal evidence.
- Add new attribution algorithms through `AttributionMethod`; do not branch on method names in `ViTExplainer`.
- Keep Captum and plotting imports lazy so core imports and `vit-explain --help` work without the explainability extra.
- Store explanation arrays as non-pickle NPZ plus deterministic JSON metadata. Do not include images or model weights.
- Run `make test-explain` after explainability changes, followed by `make check` before handoff.

## Build, Test, and Development Commands
Use `uv` and Make targets to keep local and CI behavior aligned.

- `make init`: install `uv` if missing, sync all dependency groups, and initialize the dev environment.
- `make check`: run full local gate (`style`, `quality`, `types`, `test`).
- `make style`: apply formatting/lint fixes via Ruff.
- `make quality`: run Ruff lint + format checks (no edits).
- `make types`: run static typing with `basedpyright`.
- `make test`: run pytest with coverage on `vit/`.
- `make test-ci`: run CI-equivalent tests (`not cuda and not compile`).
- `make test-compile-cpu`: run CPU `torch.compile` tests with Dynamo enabled and CUDA hidden.
- `make test-compile-cuda`: run CUDA `torch.compile` tests with Dynamo enabled.
- `make test-packed-cuda`: run packed CUDA correctness and dynamic-shape regression tests.
- `make benchmark-packed-cuda`: run three packed attention decision runs on the local CUDA GPU.
- `make test-deprecations`: run CPU tests with default deprecation warnings.
- `make audit-workflows`: audit GitHub Actions with the locked strict `zizmor` configuration.
- `make report-deprecations REPORT_DIR=<path>`: report yanked, inactive, and Python-incompatible direct pins.
- `make test-<pattern>`: run targeted tests, e.g. `make test-attention`.

GitHub Actions runs required Linux CPU checks on Python 3.11 and 3.14. The independent Monday dependency-health
workflow writes security and deprecation reports. The Sunday production workflow validates distributions and CPU
compilation. Both weekly workflows can also be dispatched manually. CUDA CI is deferred until a suitable self-hosted
runner is available; GitHub-hosted GPU larger runners are not part of the free public-repository runner allocation.

## Component Benchmark Tool
Use the local skill `$vit-component-benchmark` for detailed guidance.

- Skill file: `.agents/skills/vit-component-benchmark/SKILL.md`
- Purpose: low-level, regression-oriented benchmarking with `vit-component-benchmark`
- Includes: run/compare workflows, device/pass/dtype usage, reproducibility controls, and artifact interpretation

## Coding Style & Naming Conventions
Python target is `>=3.11,<3.15`. Keep code typed and concise.

- Formatting/linting: `ruff` (`line-length = 120`).
- Type checking: `basedpyright` (`typeCheckingMode = "standard"`).
- Naming: modules/functions/variables use `snake_case`; classes use `PascalCase`; constants use `UPPER_SNAKE_CASE`.
- Keep public APIs in `vit/` stable and explicit; add short docstrings for non-obvious behavior.

## Testing Guidelines
Use `pytest` with `pytest-cov`, `pytest-mock`, and project fixtures in `tests/conftest.py`.

- Test files should be named `test_<feature>.py`.
- Prefer parametrized tests for shape/dtype/device combinations.
- Use markers intentionally: `@pytest.mark.cuda` for GPU-required tests, `@pytest.mark.compile` for `torch.compile`.
- Run `make quality`, `make types`, and `make test-ci` before opening a PR to match required CI.
- Run `make test-compile-cpu` after changing compilation or activation-checkpointing behavior. Also run
  `make test-compile-cuda` when the changed path supports CUDA.

## Commit & Pull Request Guidelines
Recent history follows imperative, sentence-style subjects (for example: `Add ...`, `Fix ...`, `Improve ...`), often
with issue/PR refs like `(#88)`.

- Commit format: concise imperative subject; include issue reference when applicable.
- PRs should include: problem statement, behavior change summary, test evidence (commands run), and benchmark impact
  when performance-sensitive code changes.
- Ensure `make quality`, `make types`, and `make test-ci` pass before requesting review.
