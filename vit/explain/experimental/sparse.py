"""Float32 Top-K sparse autoencoders for streamed ViT residual activations."""

import heapq
import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..interventions import _site_tensor, _unpack_batch
from ..trace import preserve_explainer_state, trace_vit
from ..types import ActivationRecord, ForwardArgs, InterventionSite, TokenLayout, TraceConfig, ViTTrace


@dataclass(frozen=True)
class SparseMetrics:
    reconstruction_mse: float
    explained_variance: float
    l0: float
    dead_feature_rate: float
    downstream_score_recovery: float | None = None


@dataclass(frozen=True)
class SparseFeatureAtlas:
    top_k: int
    features: dict[int, tuple[ActivationRecord, ...]]
    layout: TokenLayout | None = None


class TopKSparseAutoencoder(nn.Module):
    """A separable Top-K dictionary with unit-norm decoded feature directions."""

    def __init__(self, input_features: int, dictionary_features: int, k: int, *, device=None):
        super().__init__()
        if input_features <= 0 or dictionary_features <= 0:
            raise ValueError("input and dictionary feature counts must be positive")
        if k <= 0 or k > dictionary_features:
            raise ValueError("k must be in [1, dictionary_features]")
        self.input_features = input_features
        self.dictionary_features = dictionary_features
        self.k = k
        self.encoder = nn.Linear(input_features, dictionary_features, dtype=torch.float32, device=device)
        self.decoder = nn.Parameter(
            torch.empty(dictionary_features, input_features, dtype=torch.float32, device=device)
        )
        self.decoder_bias = nn.Parameter(torch.zeros(input_features, dtype=torch.float32, device=device))
        self.reset_parameters()

    @property
    def decoder_directions(self) -> Tensor:
        return F.normalize(self.decoder, dim=1)

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        nn.init.normal_(self.decoder, std=self.input_features**-0.5)
        nn.init.zeros_(self.decoder_bias)
        self.normalize_decoder_()

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder.copy_(self.decoder_directions)

    def encode(self, inputs: Tensor) -> Tensor:
        if inputs.dtype != torch.float32:
            inputs = inputs.float()
        preactivations = self.encoder(inputs - self.decoder_bias)
        values, indices = preactivations.topk(self.k, dim=-1)
        values = torch.where(values == 0, torch.full_like(values, torch.finfo(values.dtype).eps), values)
        return torch.zeros_like(preactivations).scatter(-1, indices, values)

    def decode(self, codes: Tensor) -> Tensor:
        return codes @ self.decoder_directions + self.decoder_bias

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.encode(inputs)
        return self.decode(codes), codes

    def steer(self, activations: Tensor, *, feature: int, coefficient: float | Tensor) -> Tensor:
        """Add one decoded feature direction to a residual activation."""
        if feature < 0 or feature >= self.dictionary_features:
            raise ValueError(f"feature must be in [0, {self.dictionary_features})")
        coefficient_tensor = torch.as_tensor(coefficient, device=activations.device, dtype=activations.dtype)
        return activations + coefficient_tensor[..., None] * self.decoder_directions[feature].to(activations)


def sparse_metrics(
    inputs: Tensor,
    reconstruction: Tensor,
    codes: Tensor,
    *,
    downstream_score_recovery: float | None = None,
) -> SparseMetrics:
    """Measure reconstruction quality, sparsity, and dictionary utilization."""
    residual = inputs.float() - reconstruction.float()
    mse = residual.square().mean()
    variance = (inputs.float() - inputs.float().mean(dim=0, keepdim=True)).square().mean()
    explained_variance = 1 - mse / variance.clamp_min(torch.finfo(variance.dtype).eps)
    l0 = (codes != 0).sum(dim=-1).float().mean()
    dead_count = int((codes != 0).flatten(0, -2).any(dim=0).logical_not().sum().item())
    return SparseMetrics(
        reconstruction_mse=float(mse.item()),
        explained_variance=float(explained_variance.item()),
        l0=float(l0.item()),
        dead_feature_rate=dead_count / codes.shape[-1],
        downstream_score_recovery=downstream_score_recovery,
    )


def score_recovery(original: Tensor, reconstructed: Tensor, ablated: Tensor) -> Tensor:
    """Fraction of the original score effect recovered by reconstructed activations."""
    denominator = original - ablated
    epsilon = torch.finfo(denominator.dtype).eps
    safe_denominator = torch.where(denominator.abs() < epsilon, torch.full_like(denominator, epsilon), denominator)
    return (reconstructed - ablated) / safe_denominator


def _autoencoder_inputs(autoencoder: TopKSparseAutoencoder, values: Tensor) -> Tensor:
    return values.to(device=autoencoder.decoder.device, dtype=torch.float32)


def reconstruct_trace_site(
    autoencoder: TopKSparseAutoencoder,
    trace: ViTTrace,
    *,
    site: InterventionSite,
    layer: int,
) -> Tensor:
    """Reconstruct valid visual tokens while leaving prefixes and padding unchanged."""
    values = _site_tensor(trace, site, layer)
    prefix_length = trace.layout.prefix_length
    validity = trace.layout.sequence_validity

    def reconstruct_valid(visual: Tensor) -> Tensor:
        reconstructed = visual.clone()
        if validity.any():
            valid_reconstruction, _ = autoencoder(_autoencoder_inputs(autoencoder, visual[validity]))
            reconstructed[validity] = valid_reconstruction.to(visual)
        return reconstructed

    if site == "head_output":
        sequence = values.permute(0, 2, 1, 3).flatten(2)
        visual = sequence[:, prefix_length:]
        sequence = torch.cat((sequence[:, :prefix_length], reconstruct_valid(visual)), dim=1)
        return sequence.view(values.shape[0], values.shape[2], values.shape[1], values.shape[3]).permute(0, 2, 1, 3)
    visual = values[:, prefix_length:]
    return torch.cat((values[:, :prefix_length], reconstruct_valid(visual)), dim=1)


def train_sparse_autoencoder(
    autoencoder: TopKSparseAutoencoder,
    activations: Iterable[Tensor],
    *,
    steps: int,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Train on streamed activation batches without retaining the activation dataset."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    losses: list[float] = []
    iterator = iter(activations)
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(activations)
            try:
                batch = next(iterator)
            except StopIteration:
                raise ValueError("activation stream is empty") from None
        batch = _autoencoder_inputs(autoencoder, batch).reshape(-1, autoencoder.input_features)
        reconstruction, _ = autoencoder(batch)
        loss = F.mse_loss(reconstruction, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        autoencoder.normalize_decoder_()
        losses.append(float(loss.detach().item()))
    return losses


@dataclass(frozen=True)
class ViTActivationStream:
    explainer: Any
    dataloader: Iterable[Any]
    site: InterventionSite
    layer: int
    forward_args: ForwardArgs

    def __iter__(self) -> Iterator[Tensor]:
        offset = 0
        for batch in self.dataloader:
            with preserve_explainer_state(self.explainer), torch.no_grad():
                inputs, _ = _unpack_batch(batch, offset)
                trace = trace_vit(self.explainer.model, inputs, TraceConfig(), self.forward_args)
                values = _site_tensor(trace, self.site, self.layer)
                if self.site == "head_output":
                    values = values.permute(0, 2, 1, 3).flatten(2)
                visual_values = values[:, trace.layout.prefix_length :]
                values = visual_values[trace.layout.sequence_validity].float().detach()
                offset += inputs.shape[0]
            yield values


def stream_vit_activations(
    explainer,
    dataloader: Iterable[Any],
    *,
    site: InterventionSite,
    layer: int,
    forward_args: ForwardArgs | None = None,
) -> ViTActivationStream:
    """Create a restartable float32 stream over visual-token activations."""
    return ViTActivationStream(explainer, dataloader, site, layer, forward_args or ForwardArgs())


def scan_sparse_features(
    autoencoder: TopKSparseAutoencoder,
    explainer,
    dataloader: Iterable[Any],
    *,
    site: InterventionSite,
    layer: int,
    top_k: int = 10,
    forward_args: ForwardArgs | None = None,
) -> SparseFeatureAtlas:
    """Retain top activating patch coordinates for every learned sparse feature."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    arguments = forward_args or ForwardArgs()
    heaps: dict[int, list[tuple[float, int, ActivationRecord]]] = {}
    counter = itertools.count()
    offset = 0
    layout: TokenLayout | None = None
    with preserve_explainer_state(explainer), torch.no_grad():
        for batch in dataloader:
            inputs, sample_ids = _unpack_batch(batch, offset)
            trace = trace_vit(explainer.model, inputs, TraceConfig(), arguments)
            if layout is None:
                layout = trace.layout
            elif not trace.layout.spatially_matches(layout):
                raise ValueError("sparse feature atlas batches must share one token layout")
            values = _site_tensor(trace, site, layer)
            if site == "head_output":
                values = values.permute(0, 2, 1, 3).flatten(2)
            values = values[:, trace.layout.prefix_length :]
            codes = autoencoder.encode(_autoencoder_inputs(autoencoder, values))
            for batch_index, sample_id in enumerate(sample_ids):
                for token_index in range(codes.shape[1]):
                    flat_patch = int(trace.layout.visual_indices[batch_index, token_index].item())
                    if flat_patch < 0:
                        continue
                    coordinate = divmod(flat_patch, trace.layout.grid_size[1])
                    for feature, value in enumerate(codes[batch_index, token_index].tolist()):
                        record = ActivationRecord(sample_id, float(value), coordinate)
                        heap = heaps.setdefault(feature, [])
                        item = (record.value, next(counter), record)
                        if len(heap) < top_k:
                            heapq.heappush(heap, item)
                        elif item[0] > heap[0][0]:
                            heapq.heapreplace(heap, item)
            offset += inputs.shape[0]
    features = {
        feature: tuple(item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True))
        for feature, heap in heaps.items()
    }
    return SparseFeatureAtlas(top_k, features, layout)
