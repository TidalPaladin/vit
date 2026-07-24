# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `vit/` (for example `vit/vit.py`, `vit/attention.py`, `vit/transformer.py`).
The Python-first explainability toolbox lives in `vit/explain/`; its sparse autoencoder remains isolated under
`vit/explain/experimental/`.
Unit tests live in `tests/` and follow module-level coverage (`tests/test_vit.py`, `tests/test_attention.py`, etc.).  
Benchmark tooling and CLI entrypoints live in `benchmark/`; generated benchmark outputs are typically written to
`benchmark_results/`.  
Build metadata and tooling configuration are in `pyproject.toml`, `Makefile`, and `.circleci/config.yml`.

## Architecture & Core Patterns
Main flow is `Images -> PatchEmbed -> Transformer -> ViTFeatures -> Heads`.

- `ViT.forward()` returns a `ViTFeatures` dataclass (not a raw tensor).
- CLS and register tokens are optional. Pooling and classification behavior remains explicit in heads.
- Prefer config-driven construction via `ViTConfig.instantiate()` and `HeadConfig.instantiate()`.
- Use `activation_checkpointing=True` in `ViTConfig` when trading latency for lower training memory.

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
- `make test-<pattern>`: run targeted tests, e.g. `make test-attention`.

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
- Run `make test-ci` before opening a PR to match CI filtering.

## Commit & Pull Request Guidelines
Recent history follows imperative, sentence-style subjects (for example: `Add ...`, `Fix ...`, `Improve ...`), often
with issue/PR refs like `(#88)`.

- Commit format: concise imperative subject; include issue reference when applicable.
- PRs should include: problem statement, behavior change summary, test evidence (commands run), and benchmark impact
  when performance-sensitive code changes.
- Ensure `make quality`, `make types`, and `make test-ci` pass before requesting review.
