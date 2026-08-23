from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch._dynamo
from torch.testing import assert_close

import vit.attention as attention_module
import vit.vit as vit_module
from vit.attention import SelfAttention
from vit.packed import (
    PackedBatchBudget,
    PackedMemoryCalibration,
    PackedSequence,
    build_packed_batches,
    calibrate_packed_batch_budget,
    packed_configuration_fingerprint,
)
from vit.transformer import TransformerEncoderLayer, _packed_drop_path_scale
from vit.vit import PackedViTFeatures, ViT, ViTConfig, ViTFeatures


HIDDEN_SIZE = 64
NUM_HEADS = 4
LENGTHS = (7, 3, 5)


def _packed(lengths: Sequence[int], *, requires_grad: bool = False) -> PackedSequence:
    values = torch.randn(sum(lengths), HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16, requires_grad=requires_grad)
    return PackedSequence.from_lengths(values, lengths)


def _per_sequence_attention(
    module: SelfAttention, packed: PackedSequence, rope: torch.Tensor | None = None
) -> torch.Tensor:
    outputs = []
    start = 0
    for length in packed.lengths.tolist():
        end = start + length
        sequence_rope = None if rope is None else rope[:, start:end]
        if sequence_rope is not None:
            sequence_rope = sequence_rope[:, None]
        outputs.append(module(packed.values[start:end].unsqueeze(0), rope=sequence_rope).squeeze(0))
        start = end
    return torch.cat(outputs)


@pytest.mark.cuda
class TestPackedSequence:
    def test_round_trip_preserves_values_and_validity(self):
        packed = _packed(LENGTHS)

        padded, validity = packed.to_padded()
        restored = PackedSequence.from_padded(padded, validity)

        assert restored.lengths.tolist() == list(LENGTHS)
        assert restored.cu_seqlens.dtype == torch.int32
        assert restored.cu_seqlens.device.type == "cuda"
        assert restored.batch_size == len(LENGTHS)
        assert restored.min_seqlen == min(LENGTHS)
        assert restored.max_seqlen == max(LENGTHS)
        assert_close(restored.values, packed.values)

    @pytest.mark.parametrize(
        ("offsets", "message"),
        [
            ([1, 2], "start at zero"),
            ([0, 2, 2], "at least one token"),
            ([0, 3, 2], "monotonic"),
            ([0, 2], "total token count"),
        ],
    )
    def test_rejects_malformed_offsets(self, offsets, message):
        values = torch.randn(3, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        with pytest.raises(ValueError, match=message):
            PackedSequence(values, cu_seqlens)

    def test_rejects_non_cuda_or_non_int32_offsets(self):
        values = torch.randn(3, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="CUDA int32"):
            PackedSequence(values, torch.tensor([0, 3], device="cuda", dtype=torch.int64))
        with pytest.raises(ValueError, match="same device"):
            PackedSequence(values, torch.tensor([0, 3], dtype=torch.int32))

    def test_rejects_empty_batch(self):
        values = torch.empty(0, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="at least one sequence"):
            PackedSequence(values, torch.tensor([0], device="cuda", dtype=torch.int32))

    def test_validates_value_and_length_construction(self):
        values = torch.randn(3, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        offsets = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
        with pytest.raises(ValueError, match="packed values"):
            PackedSequence(values.unsqueeze(0), offsets)
        with pytest.raises(ValueError, match="one-dimensional int32 or int64"):
            PackedSequence.from_lengths(values, torch.tensor([3.0], device="cuda"))
        with pytest.raises(ValueError, match="contain integers"):
            PackedSequence.from_lengths(values, [True])

    def test_validates_padded_construction_and_replacement(self):
        padded = torch.randn(2, 3, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        validity = torch.ones(2, 3, device="cuda", dtype=torch.bool)
        with pytest.raises(ValueError, match="padded values"):
            PackedSequence.from_padded(padded.flatten(0, 1), validity)
        with pytest.raises(ValueError, match="match the padded"):
            PackedSequence.from_padded(padded, validity[:, :2])
        with pytest.raises(ValueError, match="boolean"):
            PackedSequence.from_padded(padded, validity.int())
        with pytest.raises(ValueError, match="same device"):
            PackedSequence.from_padded(padded, validity.cpu())

        packed = PackedSequence.from_padded(padded, validity)
        assert packed.jagged.values().shape == packed.values.shape
        with pytest.raises(ValueError, match="preserve"):
            packed.with_values(packed.values[:, :-1])
        with pytest.raises(ValueError, match="device"):
            packed.with_values(packed.values.cpu())


@pytest.mark.cuda
class TestPackedSelfAttention:
    @pytest.mark.parametrize("qk_normalization", [False, True])
    def test_matches_independent_sequences_and_gradients(self, qk_normalization):
        module = SelfAttention(
            HIDDEN_SIZE,
            NUM_HEADS,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            qk_normalization=qk_normalization,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        reference = SelfAttention(
            HIDDEN_SIZE,
            NUM_HEADS,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            qk_normalization=qk_normalization,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        reference.load_state_dict(module.state_dict())
        packed = _packed(LENGTHS, requires_grad=True)
        reference_values = packed.values.detach().clone().requires_grad_()
        reference_packed = PackedSequence(reference_values, packed.cu_seqlens)

        actual = module.forward_packed(packed, backend="pytorch")
        expected = _per_sequence_attention(reference, reference_packed)
        actual.values.float().square().mean().backward()
        expected.float().square().mean().backward()

        assert_close(actual.values, expected, atol=2e-2, rtol=2e-2)
        assert_close(packed.values.grad, reference_values.grad, atol=3e-2, rtol=3e-2)
        for actual_parameter, expected_parameter in zip(module.parameters(), reference.parameters(), strict=True):
            assert_close(actual_parameter.grad, expected_parameter.grad, atol=3e-2, rtol=3e-2)

    def test_rope_matches_independent_sequences(self):
        module = SelfAttention(
            HIDDEN_SIZE,
            NUM_HEADS,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        ).eval()
        reference = SelfAttention(
            HIDDEN_SIZE,
            NUM_HEADS,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        ).eval()
        reference.load_state_dict(module.state_dict())
        packed = _packed(LENGTHS)
        angles = torch.randn(1, packed.values.shape[0], HIDDEN_SIZE // NUM_HEADS, device="cuda")
        rope = torch.cat((angles.sin(), angles.cos()), dim=0)

        actual = module.forward_packed(packed, rope=rope, backend="pytorch")
        expected = _per_sequence_attention(reference, packed, rope)

        assert_close(actual.values, expected, atol=2e-2, rtol=2e-2)

    def test_training_dropout_supports_backward(self):
        module = SelfAttention(
            HIDDEN_SIZE,
            NUM_HEADS,
            hidden_dropout=0.2,
            attention_dropout=0.2,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        ).train()
        packed = _packed(LENGTHS, requires_grad=True)

        first = module.forward_packed(packed, backend="pytorch")
        second = module.forward_packed(packed, backend="pytorch")
        first.values.float().sum().backward()

        assert not torch.equal(first.values, second.values)
        assert packed.values.grad is not None

    def test_sequence_isolation(self):
        module = SelfAttention(
            HIDDEN_SIZE,
            NUM_HEADS,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        ).eval()
        packed = _packed(LENGTHS)
        perturbed_values = packed.values.clone()
        perturbed_values[: LENGTHS[0]].add_(100)

        original = module.forward_packed(packed, backend="pytorch")
        perturbed = module.forward_packed(packed.with_values(perturbed_values), backend="pytorch")

        assert_close(original.values[LENGTHS[0] :], perturbed.values[LENGTHS[0] :])

    def test_explicit_flash_attention_failure_is_actionable(self, monkeypatch):
        module = SelfAttention(HIDDEN_SIZE, NUM_HEADS, device=torch.device("cuda"), dtype=torch.bfloat16)
        monkeypatch.setattr(attention_module, "_flash_attention_available", lambda _values: False)
        with pytest.raises(RuntimeError, match="flash-attn"):
            module.forward_packed(_packed(LENGTHS), backend="flash_attention")

    def test_explicit_flash_attention_rejects_unqualified_installation(self, monkeypatch):
        module = SelfAttention(HIDDEN_SIZE, NUM_HEADS, device=torch.device("cuda"), dtype=torch.bfloat16)
        monkeypatch.setattr(attention_module, "_flash_attention_available", lambda _values: True)
        monkeypatch.setattr(attention_module, "_flash_attention_qualified", lambda _values: False)

        with pytest.raises(RuntimeError, match="has not qualified"):
            module.forward_packed(_packed(LENGTHS), backend="flash_attention")

    def test_optional_flash_attention_parity_and_backward(self):
        pytest.importorskip("flash_attn")
        module = SelfAttention(
            HIDDEN_SIZE,
            NUM_HEADS,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        ).train()
        packed = _packed(LENGTHS, requires_grad=True)

        pytorch_output = module.forward_packed(packed, backend="pytorch")
        flash_output = module._forward_packed_candidate(packed, backend="flash_attention")
        flash_output.values.float().square().mean().backward()

        assert_close(flash_output.values, pytorch_output.values, atol=2e-2, rtol=2e-2)
        assert packed.values.grad is not None

        dropout_module = SelfAttention(
            HIDDEN_SIZE,
            NUM_HEADS,
            hidden_dropout=0.2,
            attention_dropout=0.2,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        ).train()
        dropout_packed = _packed(LENGTHS, requires_grad=True)
        first = dropout_module._forward_packed_candidate(dropout_packed, backend="flash_attention")
        second = dropout_module._forward_packed_candidate(dropout_packed, backend="flash_attention")
        first.values.float().sum().backward()

        assert not torch.equal(first.values, second.values)
        assert dropout_packed.values.grad is not None


@pytest.mark.cuda
class TestPackedTransformer:
    def test_layer_matches_independent_sequences_with_layer_scale(self):
        layer = TransformerEncoderLayer(
            HIDDEN_SIZE,
            HIDDEN_SIZE * 2,
            NUM_HEADS,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            layer_scale=0.1,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        ).eval()
        reference = TransformerEncoderLayer(
            HIDDEN_SIZE,
            HIDDEN_SIZE * 2,
            NUM_HEADS,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            layer_scale=0.1,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        ).eval()
        reference.load_state_dict(layer.state_dict())
        packed = _packed(LENGTHS)

        actual = layer.forward_packed(packed, backend="pytorch")
        expected = torch.cat([reference(sequence.unsqueeze(0)).squeeze(0) for sequence in packed.unbind()])

        assert_close(actual.values, expected, atol=2e-2, rtol=2e-2)

    def test_drop_path_scale_is_constant_within_each_sequence(self):
        packed = _packed(LENGTHS)
        torch.manual_seed(123)

        scale = _packed_drop_path_scale(packed, 0.5, True).flatten()

        start = 0
        for length in LENGTHS:
            assert torch.unique(scale[start : start + length]).numel() == 1
            start += length

    def test_full_drop_path_returns_finite_zero_scale(self):
        packed = _packed(LENGTHS)

        scale = _packed_drop_path_scale(packed, 1.0, True)

        assert torch.count_nonzero(scale) == 0
        assert torch.isfinite(scale).all()


@pytest.mark.cuda
class TestPackedViT:
    @staticmethod
    def _model(**overrides) -> ViT:
        config = ViTConfig(
            in_channels=3,
            patch_size=(4, 4),
            img_size=(16, 16),
            depth=2,
            hidden_size=HIDDEN_SIZE,
            ffn_hidden_size=HIDDEN_SIZE * 2,
            num_attention_heads=NUM_HEADS,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            num_cls_tokens=1,
            num_register_tokens=2,
            pos_enc="rope",
            dtype=torch.bfloat16,
            **overrides,
        )
        return ViT(config, device=torch.device("cuda"))

    def test_forward_matches_independent_images_with_prefixes_and_rope(self):
        model = self._model(qk_normalization=True, layer_scale=0.1).eval()
        images = torch.randn(3, 3, 16, 16, device="cuda", dtype=torch.bfloat16)
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            ],
            device="cuda",
            dtype=torch.bool,
        )

        actual = model.forward_packed(images, mask, backend="pytorch")
        references = [model(images[index : index + 1], mask=mask[index : index + 1]) for index in range(3)]

        assert isinstance(actual, PackedViTFeatures)
        assert_close(
            actual.cls_tokens, torch.cat([features.cls_tokens for features in references]), atol=2e-2, rtol=2e-2
        )
        assert_close(
            actual.register_tokens,
            torch.cat([features.register_tokens for features in references]),
            atol=2e-2,
            rtol=2e-2,
        )
        assert_close(
            actual.visual_tokens.values,
            torch.cat([features.visual_tokens.squeeze(0) for features in references]),
            atol=2e-2,
            rtol=2e-2,
        )

        padded, validity = actual.to_padded()
        assert isinstance(padded, ViTFeatures)
        assert validity.sum(dim=1).tolist() == mask.sum(dim=1).tolist()

    def test_input_and_parameter_gradients_match_independent_images(self):
        model = self._model(qk_normalization=True, layer_scale=0.1).eval()
        reference = deepcopy(model)
        images = torch.randn(2, 3, 16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        reference_images = images.detach().clone().requires_grad_()
        mask = torch.tensor([[1] * 7 + [0] * 9, [1] * 13 + [0] * 3], device="cuda", dtype=torch.bool)

        actual = model.forward_packed(images, mask, backend="pytorch")
        expected = [reference(reference_images[index : index + 1], mask=mask[index : index + 1]) for index in range(2)]
        actual_values = torch.cat(
            (actual.cls_tokens.flatten(), actual.register_tokens.flatten(), actual.visual_tokens.values.flatten())
        )
        expected_values = torch.cat(
            (
                torch.cat([features.cls_tokens for features in expected]).flatten(),
                torch.cat([features.register_tokens for features in expected]).flatten(),
                torch.cat([features.visual_tokens.squeeze(0) for features in expected]).flatten(),
            )
        )
        actual_values.float().square().mean().backward()
        expected_values.float().square().mean().backward()

        assert_close(actual_values, expected_values, atol=2e-2, rtol=2e-2)
        assert_close(images.grad, reference_images.grad, atol=3e-2, rtol=3e-2)
        for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters(), strict=True):
            assert_close(actual_parameter.grad, expected_parameter.grad, atol=3e-2, rtol=3e-2)

    def test_encode_packed_matches_image_entrypoint(self):
        model = self._model().eval()
        images = torch.randn(2, 3, 16, 16, device="cuda", dtype=torch.bfloat16)
        mask = torch.tensor([[1] * 6 + [0] * 10, [1] * 12 + [0] * 4], device="cuda", dtype=torch.bool)
        visual = model.stem(images)
        packed_visual = PackedSequence.from_padded(visual, mask)
        dense_rope = model.prepare_rope((4, 4))
        packed_rope = torch.stack(
            (
                dense_rope[0].expand(images.shape[0], -1, -1)[mask],
                dense_rope[1].expand(images.shape[0], -1, -1)[mask],
            )
        )

        encoded = model.encode_packed(
            packed_visual,
            rope=packed_rope,
            tokenized_size=(4, 4),
            backend="pytorch",
        )
        forwarded = model.forward_packed(images, mask, backend="pytorch")

        assert_close(encoded.cls_tokens, forwarded.cls_tokens)
        assert_close(encoded.register_tokens, forwarded.register_tokens)
        assert_close(encoded.visual_tokens.values, forwarded.visual_tokens.values)

    def test_activation_checkpointing_supports_backward(self):
        model = self._model(activation_checkpointing=True).train()
        images = torch.randn(2, 3, 16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        mask = torch.tensor([[1] * 7 + [0] * 9, [1] * 11 + [0] * 5], device="cuda", dtype=torch.bool)

        features = model.forward_packed(images, mask, backend="pytorch")
        features.visual_tokens.values.float().square().mean().backward()

        assert images.grad is not None
        assert not images.grad.isnan().any()

    @pytest.mark.parametrize(
        "override",
        [
            {"conditioning_size": 16},
            {"specialize_global_token_norms": True},
        ],
    )
    def test_rejects_unsupported_configuration(self, override):
        model = self._model(**override)
        images = torch.randn(2, 3, 16, 16, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(2, 16, device="cuda", dtype=torch.bool)
        with pytest.raises(RuntimeError, match="does not support"):
            model.forward_packed(images, mask, backend="pytorch")

    def test_rejects_quantization_and_export(self, monkeypatch):
        model = self._model()
        images = torch.randn(2, 3, 16, 16, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(2, 16, device="cuda", dtype=torch.bool)
        model._packed_quantization_enabled = True
        with pytest.raises(RuntimeError, match="quantization"):
            model.forward_packed(images, mask, backend="pytorch")

        model._packed_quantization_enabled = False
        monkeypatch.setattr(torch.compiler, "is_exporting", lambda: True)
        with pytest.raises(RuntimeError, match="torch.export"):
            model.forward_packed(images, mask, backend="pytorch")

    def test_rejects_explainability_trace_context(self):
        model = self._model()
        images = torch.randn(2, 3, 16, 16, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(2, 16, device="cuda", dtype=torch.bool)
        trace_token = vit_module._EXPLAINABILITY_TRACE_ACTIVE.set(True)
        try:
            with pytest.raises(RuntimeError, match="explainability tracing"):
                model.forward_packed(images, mask, backend="pytorch")
        finally:
            vit_module._EXPLAINABILITY_TRACE_ACTIVE.reset(trace_token)


class TestPackedBatchBudget:
    def test_greedy_batches_respect_all_limits_and_report_fill(self):
        budget = PackedBatchBudget(max_seqlen=8, max_total_tokens=12, max_attention_work=80)

        result = build_packed_batches([7, 5, 4, 3, 2], budget)

        assert result.batches == ((7, 5), (4, 3, 2))
        assert 0 < result.average_fill <= 1
        assert 0 < result.worst_fill <= result.average_fill
        for batch in result.batches:
            budget.validate(batch)

    def test_rejects_outlier_instead_of_truncating(self):
        budget = PackedBatchBudget(max_seqlen=8, max_total_tokens=16)
        with pytest.raises(ValueError, match="maximum sequence length"):
            build_packed_batches([9], budget)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_seqlen": 0, "max_total_tokens": 1}, "max_seqlen"),
            ({"max_seqlen": 1, "max_total_tokens": 0}, "max_total_tokens"),
            ({"max_seqlen": 1, "max_total_tokens": 1, "max_attention_work": 0}, "max_attention_work"),
        ],
    )
    def test_budget_requires_positive_limits(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            PackedBatchBudget(**kwargs)

    @pytest.mark.parametrize(
        ("lengths", "message"),
        [
            ([], "at least one sequence"),
            ([0], "positive"),
            ([9], "maximum sequence length"),
            ([7, 6], "total-token limit"),
            ([7, 5], "attention-work limit"),
        ],
    )
    def test_budget_validation_rejects_each_limit(self, lengths, message):
        budget = PackedBatchBudget(max_seqlen=8, max_total_tokens=12, max_attention_work=60)
        with pytest.raises(ValueError, match=message):
            budget.validate(lengths)

    @pytest.mark.parametrize(
        ("lengths", "message"),
        [
            ([], "empty"),
            ([False], "positive integers"),
            ([5], "total-token limit"),
        ],
    )
    def test_batch_builder_rejects_invalid_collections(self, lengths, message):
        budget = PackedBatchBudget(max_seqlen=8, max_total_tokens=4, max_attention_work=16)
        with pytest.raises(ValueError, match=message):
            build_packed_batches(lengths, budget)

    def test_configuration_fingerprint_is_stable_and_sensitive(self):
        first = packed_configuration_fingerprint({"depth": 12}, "bf16", "cuda:0")
        assert first == packed_configuration_fingerprint({"depth": 12}, "bf16", "cuda:0")
        assert first != packed_configuration_fingerprint({"depth": 24}, "bf16", "cuda:0")

    def test_calibrator_binary_searches_peak_and_rejects_stale_fingerprint(self, monkeypatch):
        state = {"candidate": 0}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda _device: SimpleNamespace(total_memory=1000, name="test-gpu"),
        )
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (1000, 1000))
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 0)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: state["candidate"] * 100)

        def training_step(candidate: int) -> None:
            state["candidate"] = candidate

        calibration = calibrate_packed_batch_budget(
            training_step,
            max_seqlen=8,
            min_total_tokens=1,
            max_total_tokens=10,
            fingerprint="current",
            trials_per_candidate=2,
        )

        assert calibration.budget.max_total_tokens == 8
        assert calibration.observed_peak_bytes == 800
        assert PackedMemoryCalibration.from_json(calibration.to_json()) == calibration
        with pytest.raises(ValueError, match="stale"):
            calibration.require_fingerprint("changed")

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"device": torch.device("cpu")}, "requires an available CUDA"),
            ({"memory_fraction": 0.9}, "memory_fraction"),
            ({"trials_per_candidate": 0}, "trials_per_candidate"),
            ({"min_total_tokens": 2, "max_total_tokens": 1}, "search bounds"),
        ],
    )
    def test_calibrator_validates_configuration(self, monkeypatch, kwargs, message):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        arguments = {
            "max_seqlen": 8,
            "min_total_tokens": 1,
            "max_total_tokens": 2,
            "fingerprint": "current",
            **kwargs,
        }
        with pytest.raises((ValueError, RuntimeError), match=message):
            calibrate_packed_batch_budget(lambda _candidate: None, **arguments)

    def test_calibrator_confines_oom_and_rejects_an_unsafe_range(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(name="test-gpu"))
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (1000, 1000))
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 0)

        def out_of_memory(_candidate: int) -> None:
            raise torch.OutOfMemoryError

        with pytest.raises(RuntimeError, match="no safe packed"):
            calibrate_packed_batch_budget(
                out_of_memory,
                max_seqlen=8,
                min_total_tokens=1,
                max_total_tokens=2,
                fingerprint="current",
            )

    @pytest.mark.cuda
    def test_calibrator_measures_a_real_training_step_within_target(self):
        model = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        original_weight = model.weight.detach().clone()
        original_bias = model.bias.detach().clone()

        def training_step(candidate: int) -> None:
            with torch.no_grad():
                model.weight.copy_(original_weight)
                model.bias.copy_(original_bias)
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.randn(candidate, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
            model(inputs).float().square().mean().backward()
            optimizer.step()

        fingerprint = packed_configuration_fingerprint(model, optimizer, torch.bfloat16, torch.cuda.get_device_name())
        calibration = calibrate_packed_batch_budget(
            training_step,
            max_seqlen=32,
            min_total_tokens=8,
            max_total_tokens=32,
            fingerprint=fingerprint,
            trials_per_candidate=2,
        )

        assert calibration.budget.max_total_tokens == 32
        assert 0 < calibration.observed_peak_bytes <= calibration.target_peak_bytes
        calibration.require_fingerprint(fingerprint)


@pytest.mark.cuda
class TestPackedCompilation:
    @pytest.mark.compile
    def test_dynamic_compile_uses_one_graph_for_more_than_eight_shapes(self, monkeypatch):
        compiled_attention = torch.compile(
            attention_module._pytorch_packed_attention_impl,
            fullgraph=True,
            dynamic=True,
        )
        monkeypatch.setattr(attention_module, "_pytorch_packed_attention", compiled_attention)
        module = SelfAttention(
            HIDDEN_SIZE,
            NUM_HEADS,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        ).eval()
        shapes = (
            (3, 5),
            (2, 7, 4),
            (9, 2),
            (4, 6, 3, 2),
            (8, 5, 2),
            (3, 11),
            (7, 4, 2, 2),
            (6, 9),
            (5, 3, 8),
            (12, 2, 3),
        )
        torch._dynamo.utils.counters.clear()

        for lengths in shapes:
            output = module.forward_packed(_packed(lengths), backend="pytorch")
            assert output.values.shape == (sum(lengths), HIDDEN_SIZE)

        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1
