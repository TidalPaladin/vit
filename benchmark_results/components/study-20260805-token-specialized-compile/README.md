# Token-specialized attention compile study

Date: 2026-08-05

## Environment

- NVIDIA GeForce RTX 3090
- PyTorch 2.13.0+cu130
- Native BF16 parameters and inputs; no autocast
- Forward and backward pass
- Shape: batch 512, sequence 40, hidden size 384, 12 heads, 8 global tokens
- Training preset: 10% attention and projection dropout
- Ten warmup iterations, at least 50 samples and two seconds per trial
- Three fresh-process trials per compiler mode

## Authoritative results

| Mode | Mean latency (ms) | Mean p95 (ms) | Relative to retained |
| --- | ---: | ---: | ---: |
| Static, default compiler mode | 3.981 | 4.111 | baseline |
| Static, GEMM autotuned, no CUDA graphs | 3.823 | 3.877 | 4.0% faster, explicit opt-in |
| Eager outer graph, compiled QKV projection | 4.379 | 4.583 | 10.0% slower |
| Fully dynamic compiled wrapper | 4.880 | 5.151 | 22.6% slower |

The retained static fallback remains 13.9% slower than shared attention's 3.496 ms mean because specialization performs
separate global and visual normalization and QKV projection.

GEMM autotuning has a material one-time cost. The first search reported about 24 seconds of kernel benchmarking for
this graph, before the rest of compilation and process startup. Its 4.0% steady-state gain does not justify changing
the default. Production retains the default static compiler mode under `auto` and exposes the measured candidate as
the explicit `static_max_autotune` mode for workloads that amortize the cold-start cost.

The production dynamic path also completed a nine-batch-size sweep through batch 256 without cache exhaustion; the
batch-256 case measured 2.125 ms after the preceding shapes had exercised the shared compiler cache.

## Public API verification

After exposing the compile policy through `ViTConfig`, the same workload was repeated in three fresh processes per
public mode. Compilation was completed during ten warmup iterations before measurement.

| Public mode | Mean latency (ms) | Mean p95 (ms) | Relative to `auto` |
| --- | ---: | ---: | ---: |
| `auto` | 3.995 | 4.141 | baseline |
| `dynamic` | 4.819 | 4.943 | 20.6% slower |
| `static` | 4.017 | 4.123 | 0.6% slower |
| `static_max_autotune` | 3.859 | 3.926 | 3.4% faster |

The public `auto` result is 0.34% slower than the retained pre-API baseline of 3.981 ms, within the 5% regression
limit. Standard deviation across the three trial means was 0.006 ms for `auto`, 0.006 ms for `dynamic`, 0.012 ms for
`static`, and 0.024 ms for `static_max_autotune`. The earlier cold-start observation remains separate from these
steady-state measurements: the first autotuning search reported about 24 seconds of kernel benchmarking.

## Canonical artifacts

- `token-specialized-b512-dropout-static-max-wrapper-trial{1,2,3}`
- `token-specialized-b512-dropout-static-wrapper-default-trial{1,2,3}`
- `token-specialized-b512-dropout-eager-trial{1,2,3}`
- `token-specialized-b512-dropout-dynamic-wrapper-trial{1,2,3}`
- `shared-attention-b512-dropout-trial{1,2,3}`
- `token-specialized-dynamic-shape-sweep-auto`
- `public-{auto,dynamic,static,static_max_autotune}-trial{1,2,3}`

Other sibling directories record rejected or exploratory candidates and are not part of the authoritative comparison.
