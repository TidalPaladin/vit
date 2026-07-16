"""Causal activation interventions and dataset-level activation scans."""

import heapq
import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from .trace import preserve_explainer_state, trace_vit
from .types import (
    ActivationAtlas,
    ActivationRecord,
    ForwardArgs,
    Intervention,
    InterventionResult,
    InterventionSite,
    Target,
    TokenLayout,
    TraceConfig,
    ViTTrace,
    select_targets,
)


def _layouts_match(first: TokenLayout, second: TokenLayout) -> bool:
    return first.matches(second)


def _site_tensor(trace: ViTTrace, site: InterventionSite, layer: int) -> Tensor:
    captured = trace.layers[layer]
    return {
        "residual_pre": captured.residual_pre,
        "head_output": captured.head_outputs,
        "post_attention": captured.residual_post_attention,
        "mlp_output": captured.mlp_output,
        "residual_post": captured.residual_post,
    }[site]


def _selector(value: Sequence[int] | slice | None) -> Sequence[int] | slice:
    return slice(None) if value is None else value


def _selection_mask(size: int, selector: Sequence[int] | slice | None, device: torch.device) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[_selector(selector)] = True
    return mask


def _replacement(current: Tensor, intervention: Intervention, reference: Tensor | None) -> Tensor:
    if intervention.mode == "zero":
        return torch.zeros_like(current)
    if intervention.mode == "reference":
        if reference is None:
            raise ValueError("reference intervention requires reference_inputs")
        return reference.to(device=current.device, dtype=current.dtype)
    if intervention.value is None:
        raise ValueError(f"{intervention.mode} intervention requires value")
    value = torch.as_tensor(intervention.value, device=current.device, dtype=current.dtype)
    if intervention.mode == "mean" and value.ndim == 1:
        expand = [1] * current.ndim
        expand[-1] = value.shape[0]
        value = value.view(expand)
    return value.expand_as(current) if value.numel() != current.numel() else value.reshape_as(current)


def apply_intervention(current: Tensor, intervention: Intervention, reference: Tensor | None = None) -> Tensor:
    """Apply selectors without silently broadcasting an incompatible reference."""
    if reference is not None and reference.shape != current.shape:
        raise ValueError(f"reference activation shape {tuple(reference.shape)} does not match {tuple(current.shape)}")
    replacement = _replacement(current, intervention, reference)
    if intervention.tokens is None and intervention.channels is None and intervention.heads is None:
        return replacement
    token_mask = _selection_mask(current.shape[-2], intervention.tokens, current.device)
    channel_mask = _selection_mask(current.shape[-1], intervention.channels, current.device)
    if intervention.site == "head_output":
        head_mask = _selection_mask(current.shape[1], intervention.heads, current.device)
        selected = head_mask[None, :, None, None] & token_mask[None, None, :, None]
        selected = selected & channel_mask[None, None, None, :]
    else:
        if intervention.heads is not None:
            raise ValueError("head selectors are only valid at the head_output site")
        selected = token_mask[None, :, None] & channel_mask[None, None, :]
    return torch.where(selected, replacement, current)


def intervene(
    explainer,
    inputs: Tensor,
    target: Target | None,
    interventions: Sequence[Intervention],
    forward_args: ForwardArgs,
    reference_inputs: Tensor | None,
    reference_forward_args: ForwardArgs | None,
) -> InterventionResult:
    """Run simultaneous interventions and measure their selected-score effects."""
    requested = tuple(interventions)
    if not requested:
        raise ValueError("at least one intervention is required")
    with preserve_explainer_state(explainer), torch.no_grad():
        baseline = trace_vit(explainer.model, inputs, TraceConfig(), forward_args)
        baseline_scores = select_targets(explainer.output_fn(baseline.features), target)
        reference = None
        if any(item.mode == "reference" for item in requested):
            if reference_inputs is None:
                raise ValueError("reference intervention requires reference_inputs")
            reference = trace_vit(
                explainer.model,
                reference_inputs,
                TraceConfig(),
                reference_forward_args or forward_args,
            )
            if not _layouts_match(baseline.layout, reference.layout):
                raise ValueError("reference patching requires matching token layouts")

        by_site = {(item.site, item.layer): [] for item in requested}
        for item in requested:
            if item.layer < 0 or item.layer >= explainer.model.config.depth:
                raise ValueError(f"intervention layer must be in [0, {explainer.model.config.depth})")
            by_site[(item.site, item.layer)].append(item)

        def intervention_fn(site: str, layer: int, current: Tensor) -> Tensor:
            output = current
            for item in by_site.get((site, layer), ()):
                reference_value = None if reference is None else _site_tensor(reference, item.site, layer)
                output = apply_intervention(output, item, reference_value)
            return output

        changed = trace_vit(explainer.model, inputs, TraceConfig(), forward_args, intervention_fn)
        changed_scores = select_targets(explainer.output_fn(changed.features), target)
    absolute = changed_scores - baseline_scores
    denominator = baseline_scores.abs().clamp_min(torch.finfo(baseline_scores.dtype).eps)
    return InterventionResult(
        baseline_scores=baseline_scores.detach(),
        intervened_scores=changed_scores.detach(),
        absolute_change=absolute.detach(),
        relative_change=(absolute / denominator).detach(),
        interventions=requested,
    )


def _repeat_forward_args(forward_args: ForwardArgs, repeats: int) -> ForwardArgs:
    def repeat_batch(tensor: Tensor | None) -> Tensor | None:
        return None if tensor is None else torch.cat([tensor] * repeats, dim=0)

    return ForwardArgs(
        mask=repeat_batch(forward_args.mask),
        rope_seed=forward_args.rope_seed,
        output_norm=forward_args.output_norm,
        conditioning=repeat_batch(forward_args.conditioning),
    )


def _repeat_target(target: Target | None, repeats: int) -> Target | None:
    if isinstance(target, Tensor) and target.ndim > 0:
        return torch.cat([target] * repeats, dim=0)
    return target


def sweep_interventions(
    explainer,
    inputs: Tensor,
    target: Target | None,
    interventions: Sequence[Intervention],
    forward_args: ForwardArgs,
    reference_inputs: Tensor | None,
    reference_forward_args: ForwardArgs | None,
) -> tuple[InterventionResult, ...]:
    """Evaluate independent interventions with one shared baseline and one batched changed forward."""
    requested = tuple(interventions)
    if not requested:
        return ()
    for item in requested:
        if item.layer < 0 or item.layer >= explainer.model.config.depth:
            raise ValueError(f"intervention layer must be in [0, {explainer.model.config.depth})")

    with preserve_explainer_state(explainer), torch.no_grad():
        baseline = trace_vit(explainer.model, inputs, TraceConfig(), forward_args)
        baseline_scores = select_targets(explainer.output_fn(baseline.features), target)
        reference = None
        if any(item.mode == "reference" for item in requested):
            if reference_inputs is None:
                raise ValueError("reference intervention requires reference_inputs")
            reference = trace_vit(
                explainer.model,
                reference_inputs,
                TraceConfig(),
                reference_forward_args or forward_args,
            )
            if not _layouts_match(baseline.layout, reference.layout):
                raise ValueError("reference patching requires matching token layouts")

        batch_size = inputs.shape[0]

        def intervention_fn(site: str, layer: int, current: Tensor) -> Tensor:
            chunks = current.split(batch_size, dim=0)
            changed_chunks: list[Tensor] = []
            changed = False
            for item, chunk in zip(requested, chunks, strict=True):
                if (item.site, item.layer) != (site, layer):
                    changed_chunks.append(chunk)
                    continue
                reference_value = None if reference is None else _site_tensor(reference, item.site, layer)
                changed_chunks.append(apply_intervention(chunk, item, reference_value))
                changed = True
            return torch.cat(changed_chunks, dim=0) if changed else current

        batched_inputs = torch.cat([inputs] * len(requested), dim=0)
        changed = trace_vit(
            explainer.model,
            batched_inputs,
            TraceConfig(),
            _repeat_forward_args(forward_args, len(requested)),
            intervention_fn,
        )
        changed_scores = select_targets(
            explainer.output_fn(changed.features),
            _repeat_target(target, len(requested)),
        ).view(len(requested), batch_size)

    denominator = baseline_scores.abs().clamp_min(torch.finfo(baseline_scores.dtype).eps)
    return tuple(
        InterventionResult(
            baseline_scores=baseline_scores.detach(),
            intervened_scores=item_scores.detach(),
            absolute_change=(item_scores - baseline_scores).detach(),
            relative_change=((item_scores - baseline_scores) / denominator).detach(),
            interventions=(item,),
        )
        for item, item_scores in zip(requested, changed_scores, strict=True)
    )


def _unpack_batch(batch: Any, offset: int) -> tuple[Tensor, list[str]]:
    if isinstance(batch, Tensor):
        return batch, [str(offset + index) for index in range(batch.shape[0])]
    if isinstance(batch, Mapping):
        inputs = batch.get("inputs")
        ids = batch.get("sample_ids", batch.get("ids"))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        inputs, ids = batch[0], batch[1]
    else:
        raise ValueError("dataloader batches must be tensors, (inputs, sample_ids), or mappings")
    if not isinstance(inputs, Tensor):
        raise ValueError("dataloader batch inputs must be a Tensor")
    if ids is None:
        ids = [str(offset + index) for index in range(inputs.shape[0])]
    resolved_ids = [str(value) for value in ids]
    if len(resolved_ids) != inputs.shape[0]:
        raise ValueError("sample ID count must match batch size")
    return inputs, resolved_ids


def scan_activations(
    explainer,
    dataloader: Iterable[Any],
    *,
    site: InterventionSite,
    layer: int,
    top_k: int,
    forward_args: ForwardArgs,
    thumbnail: Callable[[Tensor, tuple[int, int]], Any] | None,
) -> ActivationAtlas:
    """Stream dataset batches and retain only top-k records per activation channel."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    heaps: dict[int, list[tuple[float, int, ActivationRecord]]] = {}
    counter = itertools.count()
    offset = 0
    atlas_layout: TokenLayout | None = None
    with preserve_explainer_state(explainer), torch.no_grad():
        for batch in dataloader:
            inputs, sample_ids = _unpack_batch(batch, offset)
            trace = trace_vit(explainer.model, inputs, TraceConfig(), forward_args)
            if atlas_layout is None:
                atlas_layout = trace.layout
            elif not trace.layout.spatially_matches(atlas_layout):
                raise ValueError("activation atlas batches must share one token layout")
            values = _site_tensor(trace, site, layer)
            if site == "head_output":
                values = values.permute(0, 2, 1, 3).flatten(2)[:, trace.layout.prefix_length :]
            else:
                values = values[:, trace.layout.prefix_length :]
            for batch_index, sample_id in enumerate(sample_ids):
                for token_index in range(values.shape[1]):
                    flat_patch = int(trace.layout.visual_indices[batch_index, token_index].item())
                    if flat_patch < 0:
                        continue
                    coordinate = divmod(flat_patch, trace.layout.grid_size[1])
                    candidates: list[tuple[int, float, list[tuple[float, int, ActivationRecord]]]] = []
                    for channel, value in enumerate(values[batch_index, token_index].tolist()):
                        numeric_value = float(value)
                        heap = heaps.setdefault(channel, [])
                        if len(heap) < top_k or numeric_value > heap[0][0]:
                            candidates.append((channel, numeric_value, heap))
                    if not candidates:
                        continue
                    patch_thumbnail = None if thumbnail is None else thumbnail(inputs[batch_index], coordinate)
                    for _channel, value, heap in candidates:
                        record = ActivationRecord(
                            sample_id=sample_id,
                            value=value,
                            patch_coordinate=coordinate,
                            thumbnail=patch_thumbnail,
                        )
                        item = (record.value, next(counter), record)
                        if len(heap) < top_k:
                            heapq.heappush(heap, item)
                        elif item[0] > heap[0][0]:
                            heapq.heapreplace(heap, item)
            offset += inputs.shape[0]
    channels = {
        channel: tuple(item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True))
        for channel, heap in heaps.items()
    }
    return ActivationAtlas(site=site, layer=layer, top_k=top_k, channels=channels, layout=atlas_layout)
