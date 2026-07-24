# ViT explainability

The `vit.explain` package supports this repository's native `ViT` on two-dimensional inputs. It records transformer
internals on an eager path while the regular fused and compiled path remains unchanged. Explanation calls use
evaluation mode and restore module training flags, parameter `requires_grad` flags, and existing gradients before
returning.

Three-dimensional ViTs are outside the stable scope and raise an actionable error. Gradient methods require a
floating-point model. Perturbation methods can run on a quantized model when its normal forward method accepts the
perturbed inputs.

## Installation

Core tracing, native attention methods, interventions, evaluation, and artifact inspection use the base install.
Captum methods and rendering use the pinned explainability extra:

```bash
uv sync --group explainability
# or
pip install "vit[explainability] @ git+https://github.com/TidalPaladin/vit.git"
```

The extra pins Captum 0.9.0, Matplotlib 3.11.0, and Pillow 12.3.0. Importing `vit.explain` and running
`vit-explain --help` do not import those optional packages.

## Citing the methods

When publishing results, cite each attribution method and evaluation metric used. Cite Captum as well when using
`Saliency`, `InputXGradient`, `IntegratedGradients`, `SmoothGrad`, or `PatchOcclusion`. The tables below map every
public method to its primary source and state where this toolbox implements an adaptation rather than an exact
reproduction. Copy-ready entries are in [`explainability-references.bib`](explainability-references.bib).

## Prediction adapter and targets

`output_fn` maps `ViTFeatures` to the downstream output tensor. This boundary preserves the model's real pooling and
head behavior.

```python
from vit.explain import ViTExplainer

explainer = ViTExplainer(
    model,
    output_fn=lambda features: classifier(features.visual_tokens.mean(dim=1)),
    output_modules=classifier,
)
```

List any modules captured by a callable `output_fn` in `output_modules`. Explanation calls put those modules in
evaluation mode and restore their training flags, parameter gradient flags, and existing gradients with the
backbone. A module passed directly as `output_fn` is registered automatically.

`ViTExplainer.from_head(model, name)` adapts `AttentivePoolHead`, `TransposedConv2dHead`, and `UpsampleHead`. A plain
`Head` requires `pool=...`. A pooling `torch.nn.Module` is registered automatically so its state is preserved:

```python
explainer = ViTExplainer.from_head(
    model,
    "linear",
    pool=lambda features: features.visual_tokens.mean(dim=1),
)
```

Targets select one scalar per example. Accepted values are:

- one integer class index shared by the batch;
- a one-dimensional tensor containing one class index per example;
- a tuple for one shared dense-output coordinate;
- a two-dimensional tensor containing one dense coordinate per example;
- a callable that accepts the output tensor and returns shape `(batch,)`.

Omit `target` only when `output_fn` already returns one scalar per example.

## Reproducing forward inputs

Pass every non-image input with `ForwardArgs`:

```python
from vit.explain import ForwardArgs

forward_args = ForwardArgs(
    mask=mask,
    rope_seed=17,
    output_norm=True,
    conditioning=conditioning,
)
```

The same values are used by tracing, attribution, intervention, and evaluation forwards. The mask uses the backbone
convention: `True` retains a visual token. `TokenLayout.visual_indices` maps the shortened model sequence back to the
full patch grid, and `visual_validity` distinguishes retained patches from masked positions. Ragged batches use `-1`
indices for padding.

CLS tokens precede register tokens, and both count toward `TokenLayout.prefix_length`. Native attribution methods
keep only visual-key columns when they return a spatial map. Register tokens can still affect an explanation through
attention composition and the downstream output.

Patch embedding crops dimensions to an integer number of patches. `TokenLayout.original_size` records the supplied
image, while `modeled_size` records the crop seen by the model. `interpolate_token_attribution` fills ignored bottom
and right borders with `NaN` so a rendered map does not imply that the model processed those pixels. An explicit
output size rescales the attribution directly to that canvas and preserves proportional ignored borders. Passing
`size=layout.modeled_size` returns only the modeled crop.

## Attribution methods

```python
from vit.explain import ForwardArgs, LeGrad

explanation = explainer.attribute(
    inputs,
    target=3,
    method=LeGrad(layers=(6, 7, 8, 9, 10, 11)),
    forward_args=ForwardArgs(mask=mask),
)
```

`Explanation.token_attributions`, `pixel_attributions`, and `layer_attributions` contain raw method outputs. Signed
gradient methods preserve negative values. Native positive-relevance methods remain nonnegative by definition.

| Method | Meaning | Main cost and sensitivity | Primary reference |
|---|---|---|---|
| `RawAttention(query=...)` | One layer's attention from selected queries to visual keys | One trace; target-independent; choose a head or average heads | [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762); interpretation caveats in [Abnar and Zuidema (2020)](https://aclanthology.org/2020.acl-main.385/) |
| `AttentionRollout(query=...)` | Residual-aware attention flow composed across layers | One trace; target-independent; query choice changes the result | [Abnar and Zuidema (2020)](https://aclanthology.org/2020.acl-main.385/) |
| `GradientAttentionRollout(query=...)` | Positive gradient-times-attention relevance composed across layers | One forward and one backward | Implementation-specific method inspired by [Chefer et al. (2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html) |
| `LeGrad()` | Positive attention gradients from intermediate post-block predictions, averaged over heads, queries, and layers | One trace plus one gradient call per selected layer | [Bousselham et al. (2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Bousselham_LeGrad_An_Explainability_Method_for_Vision_Transformers_via_Feature_Formation_ICCV_2025_paper.html) |
| `LayerGradCAM(layer=...)` | Channel-weighted post-block visual-token activations | One forward and one backward; layer choice matters | ViT token-grid adaptation of [Selvaraju et al. (2017)](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html) |
| `Saliency()` | Input gradient | One forward and one backward; signed unless `absolute=True` | [Simonyan et al. (2014)](https://arxiv.org/abs/1312.6034) |
| `InputXGradient()` | Input multiplied by its gradient | One forward and one backward; signed | Baseline in [Shrikumar et al. (2017)](https://proceedings.mlr.press/v70/shrikumar17a.html); formal analysis in [Ancona et al. (2018)](https://arxiv.org/abs/1711.06104) |
| `IntegratedGradients()` | Path integral from a baseline to the input | `n_steps` model evaluations; baseline-sensitive; signed and approximately complete | [Sundararajan et al. (2017)](https://proceedings.mlr.press/v70/sundararajan17a.html) |
| `SmoothGrad()` | Average saliency under input noise | `samples` gradient evaluations; noise-scale-sensitive and reproducible through `seed` | [Smilkov et al. (2017)](https://arxiv.org/abs/1706.03825) |
| `PatchOcclusion()` | Score change after replacing patch-sized windows | Many model evaluations; baseline-sensitive; supports models without gradients | Sliding-occlusion method from [Zeiler and Fergus (2014)](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53) |

LeGrad is the default recommendation for class-specific ViT inspection because it uses feature formation across
multiple layers. Integrated Gradients and patch occlusion are useful checks with different assumptions.

Raw attention is not causal evidence. An attention matrix records routing weights inside one forward computation; it
does not show that changing those routes changes the selected prediction. Use interventions or deletion metrics for
that question. Attention-method artifact metadata records the explicit query selector used to produce each map.

`GradientAttentionRollout` uses positive gradient times attention, residual identity, row normalization, and layer
composition. It does not implement the Deep Taylor Decomposition relevance rule in Chefer et al. `LayerGradCAM`
applies Grad-CAM's channel-weighting rule to a block's visual tokens rather than to convolutional feature maps. The
five generic input-attribution methods use [Captum](https://arxiv.org/abs/2009.07896); cite both Captum and the
original algorithm when reporting them.

[LibraGrad](https://openaccess.thecvf.com/content/CVPR2025/html/Mehri_LibraGrad_Balancing_Gradient_Flow_for_Universally_Better_Vision_Transformer_Attributions_CVPR_2025_paper.html)
and [Metric-Driven Attributions](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4e21153e79aff242492146d78d09fcdb-Abstract-Conference.html)
are candidates for later `AttributionMethod` implementations. Both require validation beyond the backward and
optimization paths in this release.

## Traces

```python
from vit.explain import TraceConfig

trace = explainer.trace(
    inputs,
    config=TraceConfig(layers=(0, 5, 11), retain_gradients=True, mlp_internals=True),
    forward_args=forward_args,
)
```

Each `LayerTrace` can contain the residual stream before the block, graph-connected attention probabilities,
per-head value outputs, the projected attention output, the post-attention residual, the MLP output, and the final
block residual. The eager path computes the block output from the captured attention probabilities. Gradients with
respect to those probabilities therefore belong to the output that the caller scores.

Set `mlp_internals=True` to populate `LayerTrace.mlp` for the selected layers. Each `MLPTrace` contains the normalized
input, raw `fc1` output, activation output, hidden tensor passed to `fc2`, and final output. GLU traces also contain
the linear and gate branches after configured limits and extra bias are applied. These branch fields are `None` for
non-GLU activations. `MLPTrace.output` and `LayerTrace.mlp_output` reference the same tensor.

The option is disabled by default, so ordinary traces keep the fused MLP path and return `LayerTrace.mlp=None`.
Traces retain their autograd graph. Enabling MLP internals retains more graph-connected tensors; release each trace
after use instead of accumulating traces in a long-running process.

## Causal interventions

Supported sites are `residual_pre`, `head_output`, `post_attention`, `mlp_output`, and `residual_post`. Select layers,
tokens, channels, and attention heads. Replacement modes are zero, constant, user-supplied mean, and reference
activation.

```python
from vit.explain import Intervention

effect = explainer.intervene(
    inputs,
    target=3,
    interventions=[
        Intervention(site="head_output", layer=8, heads=[2, 5], mode="zero"),
    ],
    forward_args=forward_args,
)

print(effect.absolute_change)
print(effect.relative_change)
```

Use `mode="reference"` and `reference_inputs=...` for activation patching. Reference and clean traces must have the
same token layout, including masks and original image size. The code rejects incompatible references instead of
broadcasting them. `explainer.sweep(...)` evaluates a list of interventions independently while sharing the clean
and reference traces; the changed examples run as one expanded batch, so sweep memory scales with the number of
interventions. Reference replacement adapts the activation-patching practice surveyed by
[Zhang and Nanda (2024)](https://arxiv.org/abs/2309.16042) from language models to ViT residual, head, attention, and
MLP sites.

## Dataset top activations

`scan_activations` streams a data loader and retains only the top records for each channel:

```python
atlas = explainer.scan_activations(
    dataloader,
    site="residual_pre",
    layer=7,
    top_k=20,
)
```

Batches can be `(inputs, sample_ids)` pairs or mappings with `inputs` and `sample_ids`. IDs are stored verbatim after
string conversion. Records contain activation values and patch coordinates. Raw images are not retained. Pass a
`thumbnail(image, patch_coordinate)` callback when the atlas should retain caller-produced thumbnails. The callback
is invoked at most once per patch that enters any channel's retained top-k records.

## Evaluation

Visual inspection is insufficient because some saliency methods remain visually plausible after model parameters
are randomized. Evaluate explanations against the behavior needed for the application.

```python
from vit.explain import Completeness, DeletionInsertion, Infidelity, SaCo

report = explainer.evaluate(
    inputs,
    explanation,
    target=3,
    metrics=[
        DeletionInsertion(steps=20),
        SaCo(groups=10),
        Infidelity(samples=32),
        Completeness(),
    ],
    forward_args=forward_args,
)
```

| Metric | Toolbox result | Primary reference |
|---|---|---|
| `DeletionInsertion` | Patch-ranked deletion and insertion curves plus area under each curve | Patch-grid adaptation of [Petsiuk et al. (2018)](https://bmva-archive.org.uk/bmvc/2018/contents/papers/1064.pdf) |
| `SaCo` | Signed pairwise concordance between group relevance and ablation-induced score change | [Wu et al. (2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_On_the_Faithfulness_of_Vision_Transformer_Explanations_CVPR_2024_paper.html) |
| `Infidelity` | Expected squared mismatch between attribution response and output response | [Yeh et al. (2019)](https://papers.nips.cc/paper_files/paper/2019/hash/a7471fdc77b3435276507cc8f2dc2569-Abstract.html) |
| `Sensitivity` | Sampled maximum attribution change under bounded Gaussian perturbations | Sampled variant of max-sensitivity from [Yeh et al. (2019)](https://papers.nips.cc/paper_files/paper/2019/hash/a7471fdc77b3435276507cc8f2dc2569-Abstract.html) |
| `Completeness` | Absolute residual between attribution sum and baseline-to-input score change | Completeness property used by [Sundararajan et al. (2017)](https://proceedings.mlr.press/v70/sundararajan17a.html) |
| `Localization` | Pointing-game hit and fraction of positive relevance inside a supplied region | [Zhang et al. (2016)](https://arxiv.org/abs/1608.00507); positive-mass form follows the energy-based pointing game used by [Wang et al. (2020)](https://openaccess.thecvf.com/content_CVPRW_2020/html/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.html) |
| `ParameterRandomizationSanity` | Centered cosine similarity after deterministic full-model randomization | Single-randomization variant of [Adebayo et al. (2018)](https://proceedings.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) |

Baseline-dependent metrics accept an explicit baseline. Metrics exclude masked and padded patches through
`TokenLayout.visual_validity`; valid patches must have finite attribution values. Evaluation rejects inputs or masks
whose token layout differs from the explanation. Report baseline choices, perturbation settings, and adaptation
details with results.

## Artifacts and CLI

```python
from vit.explain import load_explanation, save_explanation

save_explanation(explanation, "example.npz")
loaded = load_explanation("example.npz")
```

Numeric values are stored in `example.npz`; deterministic metadata is stored in `example.json`. Loading uses
`numpy.load(..., allow_pickle=False)`. BF16 tensors use exact float32 storage with dtype metadata and are restored to
BF16 when loaded. Save and load reject array shapes or token mappings that disagree with the recorded layout.
Tensor-valued method settings, such as an Integrated Gradients baseline, retain shape and dtype provenance without
retaining the tensor payload. Artifacts exclude source images and model weights.

The CLI reads artifacts. It does not load or execute models:

```bash
vit-explain --format text inspect example.npz
vit-explain --format json compare first.npz second.npz
vit-explain render example.npz heatmap.png --normalization symmetric
```

Global controls are `--format {text,json}`, `--color {auto,always,never}`, `--no-color`, mutually exclusive `--quiet`
and `--verbose`, and `--progress {auto,always,never}`. Reports go to stdout. Diagnostics go to stderr. Runtime,
artifact, optional-dependency, and I/O failures return exit code 2. `render` refuses to replace an output unless
`--overwrite` is present. `compare` requires identical token layouts, including grid geometry, prefix-token counts,
visual indices, and validity masks.

## Experimental sparse autoencoders

`vit.explain.experimental.sparse` contains a float32 Top-K sparse autoencoder for streamed residual activations. It
reports reconstruction MSE, explained variance, L0, dead-feature rate, and optional downstream score recovery. The
decoder exposes unit-normalized feature directions and decoded feature steering.

```python
from vit.explain.experimental.sparse import (
    TopKSparseAutoencoder,
    scan_sparse_features,
    stream_vit_activations,
    train_sparse_autoencoder,
)

sae = TopKSparseAutoencoder(model.config.hidden_size, 8 * model.config.hidden_size, k=32, device=inputs.device)
stream = stream_vit_activations(explainer, dataloader, site="residual_post", layer=7)
losses = train_sparse_autoencoder(sae, stream, steps=1_000)
atlas = scan_sparse_features(sae, explainer, dataloader, site="residual_post", layer=7, top_k=20)
```

All batches in one activation stream or feature scan must share the same spatial token layout.
Activation streams yield only valid visual tokens as a two-dimensional `(tokens, channels)` tensor; ragged padding
is not used to train the autoencoder. Reconstruction likewise changes only valid visual tokens and leaves prefix and
padding positions unchanged. Training, feature scans, and reconstruction transfer activations to the autoencoder's
device; reconstructed trace tensors return to the trace device.

This namespace is experimental and may change independently of the stable toolbox. Interpretable feature claims
need top-activation inspection, decoded steering, and downstream score recovery, not reconstruction error alone.
The Top-K training design follows [Gao et al. (2024)](https://arxiv.org/abs/2406.04093). ViT-specific evidence comes
from [Kim et al. (2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/50cf815fac839ac68846304ea1613aaa-Abstract-Conference.html),
which applies sparse autoencoders to ViT residual streams. Broader mechanistic context includes
[Anthropic's circuit-tracing work](https://www.anthropic.com/research/tracing-thoughts-language-model) and
[OpenAI's sparse-feature analysis](https://openai.com/index/extracting-concepts-from-gpt-4/).

## Validation and performance measurement

Run the focused explainability suite, then the repository gates:

```bash
make test-explain
make check
make test-ci
```

Use `examples/benchmark_explainability.py` to measure tracing, LeGrad, Integrated Gradients, patch occlusion, and a
causal sweep on the current hardware. The script reports latency and CUDA peak memory for a tiny fixture and a
ViT-B-shaped fixture. Results are descriptive; the repository does not impose hardware-specific thresholds.

The following development measurement used an NVIDIA GeForce RTX 3090 on 2026-07-15, eager execution, batch size 1,
one measured repeat, and four Integrated Gradients steps. The ViT-B-shaped fixture used 12 layers, width 768, 12
heads, 224 by 224 inputs, and 16 by 16 patches.

| Fixture | Method | Latency (ms) | CUDA peak (MiB) |
|---|---|---:|---:|
| Tiny | Trace | 2.57 | 8.6 |
| Tiny | LeGrad | 4.87 | 16.9 |
| Tiny | Integrated Gradients | 10.88 | 17.9 |
| Tiny | Patch occlusion | 118.84 | 17.0 |
| Tiny | Four-item causal sweep | 9.44 | 17.2 |
| ViT-B-shaped | Trace | 11.21 | 517.9 |
| ViT-B-shaped | LeGrad | 27.63 | 530.0 |
| ViT-B-shaped | Integrated Gradients | 47.03 | 1025.6 |
| ViT-B-shaped | Patch occlusion | 1462.89 | 629.8 |
| ViT-B-shaped | Four-item causal sweep | 89.30 | 695.0 |

Run at least three repeats and use the intended Integrated Gradients step count before drawing performance
conclusions for a deployment model.
