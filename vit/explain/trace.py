"""Graph-connected eager tracing for native two-dimensional ViTs."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vit.attention import _permute_and_fold_head, project_qkv_packed
from vit.norm import get_norm_bias
from vit.patch_embed import PatchEmbed2d
from vit.tokens import apply_mask
from vit.transformer import _forward_mlp
from vit.vit import ViT, ViTFeatures

from .types import ForwardArgs, LayerTrace, TokenLayout, TraceConfig, ViTTrace


class _StatefulExplainer(Protocol):
    model: ViT
    output_modules: tuple[nn.Module, ...]


@contextmanager
def preserve_model_state(*models: nn.Module) -> Iterator[None]:
    """Run explanation code in eval mode without changing caller-owned module state."""
    training_states = {module: module.training for model in models for module in model.modules()}
    parameter_states = {
        parameter: (parameter.requires_grad, None if parameter.grad is None else parameter.grad.detach().clone())
        for model in models
        for parameter in model.parameters()
    }
    for model in models:
        model.eval()
    try:
        yield
    finally:
        for module, training in training_states.items():
            module.training = training
        for parameter, (requires_grad, gradient) in parameter_states.items():
            parameter.requires_grad_(requires_grad)
            parameter.grad = gradient


@contextmanager
def preserve_explainer_state(explainer: _StatefulExplainer) -> Iterator[None]:
    """Preserve the backbone and any external modules used by ``output_fn``."""
    with preserve_model_state(explainer.model, *explainer.output_modules):
        yield


def _visual_indices(mask: Tensor | None, batch_size: int, token_count: int, device: torch.device) -> Tensor:
    indices = torch.arange(token_count, device=device).view(1, token_count, 1).expand(batch_size, -1, -1)
    if mask is None:
        return indices.squeeze(-1)
    return apply_mask(mask, indices, padding_value=-1).squeeze(-1)


def make_token_layout(model: ViT, inputs: Tensor, mask: Tensor | None) -> TokenLayout:
    """Describe prefix, masked, padded, and ignored-border token semantics."""
    if inputs.ndim != 4 or len(model.config.patch_size) != 2:
        raise ValueError("vit.explain supports 2D ViT inputs with shape (batch, channels, height, width)")
    if not isinstance(model.stem, PatchEmbed2d):
        raise ValueError("vit.explain supports 2D ViT inputs with shape (batch, channels, height, width)")
    grid_size = model.stem.tokenized_size(inputs.shape[2:])
    token_count = grid_size[0] * grid_size[1]
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != (inputs.shape[0], token_count):
            raise ValueError(f"mask must be bool with shape {(inputs.shape[0], token_count)}, got {tuple(mask.shape)}")
        mask = mask.to(device=inputs.device)
        validity = mask
    else:
        validity = torch.ones((inputs.shape[0], token_count), dtype=torch.bool, device=inputs.device)
    patch_size = tuple(model.config.patch_size)
    return TokenLayout(
        grid_size=grid_size,
        patch_size=(int(patch_size[0]), int(patch_size[1])),
        original_size=(int(inputs.shape[-2]), int(inputs.shape[-1])),
        modeled_size=model.stem.original_size(grid_size),
        num_cls_tokens=model.config.num_cls_tokens,
        num_register_tokens=model.config.num_register_tokens,
        visual_indices=_visual_indices(mask, inputs.shape[0], token_count, inputs.device),
        visual_validity=validity,
    )


def attention_output_from_heads(attention, head_outputs: Tensor) -> Tensor:
    """Project per-head values back into the residual stream."""
    output = F.linear(_permute_and_fold_head(head_outputs), attention.out_proj.weight, attention.out_proj.bias)
    return attention.dropout(output)


def eager_self_attention(attention, x: Tensor, rope: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    """Compute attention output from explicitly materialized probabilities."""
    q_norm_weight = attention.q_norm.weight if attention.q_norm is not None else None
    q_norm_bias = get_norm_bias(attention.q_norm) if attention.q_norm is not None else None
    k_norm_weight = attention.k_norm.weight if attention.k_norm is not None else None
    k_norm_bias = get_norm_bias(attention.k_norm) if attention.k_norm is not None else None
    qk_eps = attention.q_norm.eps or 1e-5 if attention.q_norm is not None else 1e-5
    q, k, value = project_qkv_packed(
        x,
        attention.qkv_proj.weight,
        attention.qkv_proj.bias,
        attention.norm.weight,
        get_norm_bias(attention.norm),
        attention._use_layer_norm,
        attention._head_dim,
        attention.norm.eps or 1e-5,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        attention._use_layer_norm,
        qk_eps,
        attention._qk_normalization,
        rope,
    )
    probabilities = (q @ k.mT * (attention._head_dim**-0.5)).softmax(dim=-1)
    if not probabilities.requires_grad:
        probabilities.requires_grad_()
    head_outputs = probabilities @ value
    return attention_output_from_heads(attention, head_outputs), probabilities, head_outputs


def trace_vit(
    model: ViT,
    inputs: Tensor,
    config: TraceConfig,
    forward_args: ForwardArgs,
    intervention: Callable[[str, int, Tensor], Tensor] | None = None,
) -> ViTTrace:
    """Execute the model through its eager, inspectable transformer path."""
    layout = make_token_layout(model, inputs, forward_args.mask)
    model._validate_conditioning(forward_args.conditioning)
    tokenized_size = layout.grid_size
    x = model.normalize_patch_embeddings(model.stem(inputs))
    if forward_args.mask is not None:
        x = apply_mask(forward_args.mask.to(inputs.device), x)
    x = model.add_prefix_tokens(x)
    rope = (
        model.prepare_rope(tokenized_size, forward_args.mask, forward_args.rope_seed)
        if model.rope is not None
        else None
    )
    selected_layers = set(range(model.config.depth) if config.layers is None else config.layers)
    if any(layer < 0 or layer >= model.config.depth for layer in selected_layers):
        raise ValueError(f"trace layers must be in [0, {model.config.depth})")

    layers: list[LayerTrace] = []
    for layer_index in range(model.config.depth):
        block = model.get_block(layer_index)
        residual_pre = intervention("residual_pre", layer_index, x) if intervention is not None else x
        x = residual_pre
        attention_output, probabilities, head_outputs = eager_self_attention(block.self_attention, x, rope)
        if intervention is not None:
            head_outputs = intervention("head_output", layer_index, head_outputs)
            attention_output = attention_output_from_heads(block.self_attention, head_outputs)
        residual_post_attention = x + block.layer_scale_attn(attention_output)
        if intervention is not None:
            residual_post_attention = intervention("post_attention", layer_index, residual_post_attention)
        mlp_output = _forward_mlp(
            block.mlp,
            residual_post_attention,
            forward_args.conditioning,
            None,
            inputs.shape[0],
        )
        if intervention is not None:
            mlp_output = intervention("mlp_output", layer_index, mlp_output)
        x = residual_post_attention + block.layer_scale_mlp(mlp_output)
        if intervention is not None:
            x = intervention("residual_post", layer_index, x)
        if layer_index in selected_layers:
            layer_trace = LayerTrace(
                layer=layer_index,
                residual_pre=residual_pre,
                attention_probabilities=probabilities,
                head_outputs=head_outputs,
                attention_output=attention_output,
                residual_post_attention=residual_post_attention,
                mlp_output=mlp_output,
                residual_post=x,
            )
            if config.retain_gradients:
                for tensor in (
                    layer_trace.residual_pre,
                    layer_trace.attention_probabilities,
                    layer_trace.head_outputs,
                    layer_trace.attention_output,
                    layer_trace.residual_post_attention,
                    layer_trace.mlp_output,
                    layer_trace.residual_post,
                ):
                    if tensor.requires_grad:
                        tensor.retain_grad()
            layers.append(layer_trace)
    if forward_args.output_norm:
        x = model.output_norm(x)
    features = ViTFeatures(x, model.config.num_register_tokens, model.config.num_cls_tokens, tokenized_size)
    return ViTTrace(features=features, layout=layout, layers=tuple(layers), forward_args=forward_args)
