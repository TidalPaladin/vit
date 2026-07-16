# Codebase correctness review notes

This file records cross-cutting correctness observations found during review. It intentionally omits defects that
apply to only one method or test fixture.

## Validate compatibility at public API boundaries

APIs that combine model inputs with derived objects should verify semantic compatibility before computing a result.
Shape checks alone do not detect mismatched image geometry, token masks, prefix-token counts, or modeled crops.
Centralizing these checks around the existing token-layout representation would give attribution, evaluation,
visualization, intervention, and artifact paths the same failure behavior.

## Keep padding validity attached to sequence tensors

Ragged masks create padded sequence positions that have the same tensor shape and dtype as modeled tokens. Any code
that reconstructs, aggregates, scans, or replaces sequence activations should consume the corresponding validity
mask. A shared helper for selecting valid visual tokens and restoring them to sequence form would reduce the chance
that a new analysis path treats padding as model data. When token data is projected into pixel space, propagate the
same validity through masked patches and ignored image borders before aggregating it.

## Define one dataset batch protocol for auxiliary model inputs

Dataset-level tools need sample IDs and may also need per-batch masks, conditioning tensors, or deterministic forward
arguments. A typed batch adapter should extract these fields together instead of passing one fixed `ForwardArgs`
object across an entire loader. This would keep activation scans and future streaming analyses aligned with ordinary
model forwards when auxiliary inputs vary by example.

## Separate spatial geometry from batch-specific token validity

Grid size, patch size, modeled crop, and original image size are batch-invariant for many scans. Visual indices and
validity masks are batch-specific. Storing both concerns in one layout object is useful for a single explanation, but
dataset summaries should retain shared geometry separately from the first batch's token mapping.

## Preserve provenance when tensor payloads are omitted

Some tensor-valued settings are too large or sensitive to store in routine artifacts. A metadata sanitizer should
still record that the tensor was supplied, along with safe fields such as shape, dtype, and a caller-provided label.
Silently deleting the setting makes two computations with different inputs look as if they used the same
configuration.

## Register stateful modules accepted by convenience adapters

An adapter that accepts an arbitrary callable should detect `torch.nn.Module` instances and include them in the same
state-preservation scope as the model. Otherwise repeated analysis forwards can run an external module in training
mode or update its buffers even though the surrounding API promises deterministic evaluation behavior.

## Normalize parser failures at trust boundaries

File-format libraries often raise exceptions outside the small set documented by their top-level load function.
Artifact and CLI boundaries should translate the parser's complete malformed-input exception family into one stable,
user-facing error and exit status. Corrupt archives are useful fixtures because they exercise failures before schema
validation begins.

## Convert storage dtypes at external library boundaries

Internal tensor and artifact formats can support dtypes that downstream libraries do not. Before passing arrays to
NumPy, plotting libraries, image encoders, or other consumers, convert them to an explicitly supported interchange
dtype. Round-trip tests alone do not cover this boundary; each external consumer needs a fixture for every promised
storage dtype.

## Make cross-module device ownership explicit

Helpers that combine independently constructed PyTorch modules cannot assume their parameters share a device.
Choose which module owns the computation, transfer inputs at that boundary, and return outputs on the device required
by the caller. Cover both the common same-device path and at least one deliberate mismatch in tests.

## Record provenance and implementation differences for named algorithms

An API that uses a published algorithm's name can still differ in its propagation rule, spatial domain, sampling
scheme, or randomization procedure. Documentation and artifact metadata should record both the primary source and the
local differences so downstream users do not report an adaptation as an exact reproduction. Machine-readable method
metadata should include a stable variant identifier when two implementations share the same broad method name.
