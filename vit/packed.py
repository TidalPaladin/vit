"""Packed variable-length sequence containers and memory budgeting utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Self

import torch
from torch import Tensor


PackedAttentionBackend = Literal["auto", "pytorch", "flash_attention"]


class PackedSequence:
    """A batch of non-empty sequences stored as contiguous token values.

    ``cu_seqlens`` contains CUDA ``int32`` cumulative offsets. The underlying
    jagged NestedTensor is constructed lazily because ordinary packed-path
    pointwise operations run on ``values``.
    """

    __slots__ = ("_cu_seqlens", "_lengths", "_max_seqlen", "_min_seqlen", "_values")

    def __init__(self, values: Tensor, cu_seqlens: Tensor):
        self._validate(values, cu_seqlens)
        self._values = values
        self._cu_seqlens = cu_seqlens
        self._lengths = cu_seqlens.diff()
        self._min_seqlen = int(self._lengths.min().item())
        self._max_seqlen = int(self._lengths.max().item())

    @classmethod
    def _from_validated(
        cls,
        values: Tensor,
        cu_seqlens: Tensor,
        lengths: Tensor,
        min_seqlen: int,
        max_seqlen: int,
    ) -> Self:
        packed = cls.__new__(cls)
        packed._values = values
        packed._cu_seqlens = cu_seqlens
        packed._lengths = lengths
        packed._min_seqlen = min_seqlen
        packed._max_seqlen = max_seqlen
        return packed

    @staticmethod
    def _validate(values: Tensor, cu_seqlens: Tensor) -> None:
        if values.ndim != 2:
            raise ValueError(f"packed values must have shape [total_tokens, hidden_size], got {tuple(values.shape)}")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be one-dimensional and describe at least one sequence")
        if values.device != cu_seqlens.device:
            raise ValueError("packed values and cu_seqlens must use the same device")
        if cu_seqlens.device.type != "cuda" or cu_seqlens.dtype != torch.int32:
            raise ValueError("cu_seqlens must be a CUDA int32 tensor")
        if int(cu_seqlens[0].item()) != 0:
            raise ValueError("cu_seqlens must start at zero")
        lengths = cu_seqlens.diff()
        if bool((lengths < 0).any().item()):
            raise ValueError("cu_seqlens must be monotonic")
        if bool((lengths == 0).any().item()):
            raise ValueError("each packed sequence must contain at least one token")
        if int(cu_seqlens[-1].item()) != values.shape[0]:
            raise ValueError("cu_seqlens final offset must equal the total token count")

    @classmethod
    def from_lengths(cls, values: Tensor, lengths: Sequence[int] | Tensor) -> Self:
        """Construct a packed batch from values and per-sequence lengths."""
        if isinstance(lengths, Tensor):
            if lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError("lengths must be a one-dimensional int32 or int64 tensor")
            length_tensor = lengths.to(device=values.device, dtype=torch.int32)
        else:
            normalized_lengths = tuple(lengths)
            if any(not isinstance(length, int) or isinstance(length, bool) for length in normalized_lengths):
                raise ValueError("lengths must contain integers")
            length_tensor = torch.tensor(normalized_lengths, device=values.device, dtype=torch.int32)
        zero = torch.zeros(1, device=values.device, dtype=torch.int32)
        return cls(values, torch.cat((zero, length_tensor.cumsum(0, dtype=torch.int32))))

    @classmethod
    def from_padded(cls, values: Tensor, validity: Tensor) -> Self:
        """Remove padding from a dense batch using a ``True`` token-validity mask."""
        if values.ndim != 3:
            raise ValueError(f"padded values must have shape [batch, sequence, hidden], got {tuple(values.shape)}")
        if validity.shape != values.shape[:2]:
            raise ValueError("validity must match the padded batch and sequence dimensions")
        if validity.dtype != torch.bool:
            raise ValueError("validity must be a boolean tensor")
        if validity.device != values.device:
            raise ValueError("padded values and validity must use the same device")
        lengths = validity.sum(dim=1, dtype=torch.int32)
        return cls.from_lengths(values[validity], lengths)

    @property
    def values(self) -> Tensor:
        return self._values

    @property
    def cu_seqlens(self) -> Tensor:
        return self._cu_seqlens

    @property
    def lengths(self) -> Tensor:
        return self._lengths

    @property
    def batch_size(self) -> int:
        return self._cu_seqlens.shape[0] - 1

    @property
    def min_seqlen(self) -> int:
        return self._min_seqlen

    @property
    def max_seqlen(self) -> int:
        return self._max_seqlen

    @property
    def jagged(self) -> Tensor:
        """Return a jagged NestedTensor view over the packed values."""
        return torch.nested.nested_tensor_from_jagged(
            self.values,
            self.cu_seqlens,
            min_seqlen=self.min_seqlen,
            max_seqlen=self.max_seqlen,
        )

    def with_values(self, values: Tensor) -> Self:
        """Reuse this sequence layout with replacement values."""
        if values.shape != self.values.shape:
            raise ValueError("replacement values must preserve the packed value shape")
        if values.device != self.values.device:
            raise ValueError("replacement values must remain on the packed sequence device")
        return self._from_validated(
            values,
            self.cu_seqlens,
            self.lengths,
            self.min_seqlen,
            self.max_seqlen,
        )

    def unbind(self) -> tuple[Tensor, ...]:
        """Return ordinary tensor views for each logical sequence."""
        offsets = self.cu_seqlens.tolist()
        return tuple(self.values[start:end] for start, end in zip(offsets[:-1], offsets[1:], strict=True))

    def to_padded(self, padding_value: float = 0.0) -> tuple[Tensor, Tensor]:
        """Return dense values and a boolean validity mask without implicit consumers."""
        token_indices = torch.arange(self.max_seqlen, device=self.values.device)
        validity = token_indices.unsqueeze(0) < self.lengths.unsqueeze(1)
        padded = self.values.new_full((self.batch_size, self.max_seqlen, self.values.shape[-1]), padding_value)
        padded[validity] = self.values
        return padded, validity


@dataclass(frozen=True)
class PackedBatchBudget:
    """Hard limits checked before a packed attention kernel is launched."""

    max_seqlen: int
    max_total_tokens: int
    max_attention_work: int | None = None

    def __post_init__(self) -> None:
        if self.max_seqlen <= 0:
            raise ValueError("max_seqlen must be positive")
        if self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if self.max_attention_work is not None and self.max_attention_work <= 0:
            raise ValueError("max_attention_work must be positive when provided")

    def validate(self, lengths: Sequence[int] | Tensor) -> None:
        normalized = _normalize_lengths(lengths)
        if not normalized:
            raise ValueError("a packed batch must contain at least one sequence")
        if min(normalized) <= 0:
            raise ValueError("packed sequence lengths must be positive")
        if max(normalized) > self.max_seqlen:
            raise ValueError(f"packed batch exceeds maximum sequence length: {max(normalized)} > {self.max_seqlen}")
        total_tokens = sum(normalized)
        if total_tokens > self.max_total_tokens:
            raise ValueError(f"packed batch exceeds total-token limit: {total_tokens} > {self.max_total_tokens}")
        attention_work = sum(length * length for length in normalized)
        if self.max_attention_work is not None and attention_work > self.max_attention_work:
            raise ValueError(f"packed batch exceeds attention-work limit: {attention_work} > {self.max_attention_work}")


@dataclass(frozen=True)
class PackedBatchConstruction:
    """Greedy batches plus observable token-budget utilization."""

    batches: tuple[tuple[int, ...], ...]
    average_fill: float
    worst_fill: float


def build_packed_batches(lengths: Iterable[int], budget: PackedBatchBudget) -> PackedBatchConstruction:
    """Greedily group lengths without ever exceeding ``budget``."""
    batches: list[tuple[int, ...]] = []
    current: list[int] = []
    current_tokens = 0
    current_work = 0

    for length in lengths:
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError("packed sequence lengths must be positive integers")
        if length > budget.max_seqlen:
            raise ValueError(f"packed batch exceeds maximum sequence length: {length} > {budget.max_seqlen}")
        length_work = length * length
        if length > budget.max_total_tokens:
            raise ValueError(f"packed batch exceeds total-token limit: {length} > {budget.max_total_tokens}")
        if budget.max_attention_work is not None and length_work > budget.max_attention_work:
            raise ValueError(f"packed batch exceeds attention-work limit: {length_work} > {budget.max_attention_work}")

        exceeds_tokens = current_tokens + length > budget.max_total_tokens
        exceeds_work = budget.max_attention_work is not None and current_work + length_work > budget.max_attention_work
        if current and (exceeds_tokens or exceeds_work):
            batches.append(tuple(current))
            current = [length]
            current_tokens = length
            current_work = length_work
        else:
            current.append(length)
            current_tokens += length
            current_work += length_work
    if current:
        batches.append(tuple(current))
    if not batches:
        raise ValueError("cannot construct packed batches from an empty length collection")

    fills = tuple(_budget_fill(batch, budget) for batch in batches)
    return PackedBatchConstruction(
        batches=tuple(batches),
        average_fill=sum(fills) / len(fills),
        worst_fill=min(fills),
    )


def _normalize_lengths(lengths: Sequence[int] | Tensor) -> tuple[int, ...]:
    if isinstance(lengths, Tensor):
        return tuple(int(length) for length in lengths.tolist())
    return tuple(int(length) for length in lengths)


def _budget_fill(lengths: Sequence[int], budget: PackedBatchBudget) -> float:
    token_fill = sum(lengths) / budget.max_total_tokens
    if budget.max_attention_work is None:
        return token_fill
    work_fill = sum(length * length for length in lengths) / budget.max_attention_work
    return max(token_fill, work_fill)


@dataclass(frozen=True)
class PackedMemoryCalibration:
    """Device-bound result from disposable packed training-step trials."""

    budget: PackedBatchBudget
    fingerprint: str
    device_name: str
    usable_memory_bytes: int
    target_peak_bytes: int
    observed_peak_bytes: int
    memory_fraction: float
    trials_per_candidate: int
    calibrated_at: str

    def require_fingerprint(self, fingerprint: str) -> None:
        if fingerprint != self.fingerprint:
            raise ValueError("packed memory budget fingerprint is stale; recalibrate before use")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        data = json.loads(payload)
        data["budget"] = PackedBatchBudget(**data["budget"])
        return cls(**data)


def packed_configuration_fingerprint(*parts: Any) -> str:
    """Create a stable fingerprint for model, optimizer, precision, and device state."""
    encoded = json.dumps(parts, default=str, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def calibrate_packed_batch_budget(
    training_step: Callable[[int], None],
    *,
    max_seqlen: int,
    min_total_tokens: int,
    max_total_tokens: int,
    fingerprint: str,
    max_attention_work: int | None = None,
    memory_fraction: float = 0.85,
    trials_per_candidate: int = 3,
    device: torch.device | None = None,
) -> PackedMemoryCalibration:
    """Binary-search a safe total-token limit with disposable real-step trials.

    ``training_step`` receives the proposed total-token count. It must construct
    the caller's representative worst-case batch and restore any model or
    optimizer state that the disposable trial mutates. OOM recovery is confined
    to this calibration function and is never used by production execution.
    """
    resolved_device = torch.device("cuda") if device is None else device
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("packed memory calibration requires an available CUDA device")
    if not 0 < memory_fraction <= 0.85:
        raise ValueError("memory_fraction must be in (0, 0.85] to preserve at least a 15% reserve")
    if trials_per_candidate <= 0:
        raise ValueError("trials_per_candidate must be positive")
    if not 0 < min_total_tokens <= max_total_tokens:
        raise ValueError("token search bounds must satisfy 0 < min_total_tokens <= max_total_tokens")

    properties = torch.cuda.get_device_properties(resolved_device)
    free_memory_bytes, _ = torch.cuda.mem_get_info(resolved_device)
    current_reserved_bytes = torch.cuda.memory_reserved(resolved_device)
    usable_memory_bytes = int(free_memory_bytes + current_reserved_bytes)
    target_peak_bytes = int(usable_memory_bytes * memory_fraction)
    low = min_total_tokens
    high = max_total_tokens
    best: tuple[int, int] | None = None

    while low <= high:
        candidate = (low + high) // 2
        candidate_peak = 0
        candidate_succeeded = True
        for _ in range(trials_per_candidate):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(resolved_device)
            try:
                training_step(candidate)
                torch.cuda.synchronize(resolved_device)
            except torch.OutOfMemoryError:
                candidate_succeeded = False
                torch.cuda.empty_cache()
                break
            candidate_peak = max(candidate_peak, torch.cuda.max_memory_reserved(resolved_device))
            if candidate_peak > target_peak_bytes:
                candidate_succeeded = False
                break

        if candidate_succeeded:
            best = (candidate, candidate_peak)
            low = candidate + 1
        else:
            high = candidate - 1

    if best is None:
        raise RuntimeError("no safe packed total-token budget was found in the requested calibration range")
    safe_total_tokens, observed_peak_bytes = best
    return PackedMemoryCalibration(
        budget=PackedBatchBudget(max_seqlen, safe_total_tokens, max_attention_work),
        fingerprint=fingerprint,
        device_name=properties.name,
        usable_memory_bytes=usable_memory_bytes,
        target_peak_bytes=target_peak_bytes,
        observed_peak_bytes=observed_peak_bytes,
        memory_fraction=memory_fraction,
        trials_per_candidate=trials_per_candidate,
        calibrated_at=datetime.now(UTC).isoformat(),
    )
