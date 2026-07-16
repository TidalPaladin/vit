import pytest
import torch
from torch.testing import assert_close

from vit import ViT, ViTConfig
from vit.explain import ForwardArgs, Intervention, ViTExplainer
from vit.explain.experimental.sparse import (
    TopKSparseAutoencoder,
    reconstruct_trace_site,
    scan_sparse_features,
    score_recovery,
    sparse_metrics,
    stream_vit_activations,
    train_sparse_autoencoder,
)


def test_topk_sparse_autoencoder_has_exact_sparsity_and_normalized_decoder() -> None:
    model = TopKSparseAutoencoder(input_features=6, dictionary_features=12, k=3)
    inputs = torch.randn(5, 6)

    reconstruction, codes = model(inputs)
    model.normalize_decoder_()

    assert reconstruction.shape == inputs.shape
    assert (codes != 0).sum(1).eq(3).all()
    assert_close(model.decoder_directions.norm(dim=1), torch.ones(12))


def test_sparse_metrics_account_for_dead_features_and_reconstruction() -> None:
    inputs = torch.randn(8, 4)
    reconstruction = inputs.clone()
    codes = torch.zeros(8, 6)
    codes[:, :2] = 1

    metrics = sparse_metrics(inputs, reconstruction, codes)

    assert metrics.reconstruction_mse == 0
    assert metrics.explained_variance == 1
    assert metrics.l0 == 2
    assert metrics.dead_feature_rate == 4 / 6


def test_decoded_feature_steering_follows_decoder_direction() -> None:
    model = TopKSparseAutoencoder(input_features=4, dictionary_features=8, k=2)
    model.normalize_decoder_()
    activations = torch.zeros(3, 4)

    steered = model.steer(activations, feature=3, coefficient=2.5)

    expected = 2.5 * model.decoder_directions[3]
    assert_close(steered, expected.expand_as(steered))


def test_reconstructed_residual_can_be_injected_for_score_recovery() -> None:
    config = ViTConfig(
        in_channels=1,
        patch_size=(2, 2),
        img_size=(4, 4),
        depth=1,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pos_enc="none",
        dtype=torch.float32,
    )
    model = ViT(config).eval()
    inputs = torch.randn(2, 1, 4, 4)
    explainer = ViTExplainer(model, lambda features: features.visual_tokens.mean((1, 2)))
    trace = explainer.trace(inputs)
    autoencoder = TopKSparseAutoencoder(8, 16, k=4)
    reconstructed = reconstruct_trace_site(autoencoder, trace, site="residual_post", layer=0)

    result = explainer.intervene(
        inputs,
        target=None,
        interventions=[
            Intervention(site="residual_post", layer=0, mode="constant", value=reconstructed),
        ],
    )

    assert reconstructed.shape == trace.layers[0].residual_post.shape
    assert torch.isfinite(result.intervened_scores).all()

    head_autoencoder = TopKSparseAutoencoder(8, 16, k=4)
    reconstructed_heads = reconstruct_trace_site(head_autoencoder, trace, site="head_output", layer=0)
    assert reconstructed_heads.shape == trace.layers[0].head_outputs.shape


@pytest.mark.parametrize("site", ["residual_pre", "head_output"])
def test_reconstruction_preserves_ragged_padding(site) -> None:
    config = ViTConfig(
        in_channels=1,
        patch_size=(2, 2),
        img_size=(4, 4),
        depth=1,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pos_enc="none",
        dtype=torch.float32,
    )
    model = ViT(config).eval()
    inputs = torch.randn(2, 1, 4, 4)
    mask = torch.tensor([[True, True, True, False], [False, True, False, False]])
    explainer = ViTExplainer(model, lambda features: features.visual_tokens.mean((1, 2)))
    trace = explainer.trace(inputs, forward_args=ForwardArgs(mask=mask))
    autoencoder = TopKSparseAutoencoder(8, 16, k=4)
    with torch.no_grad():
        autoencoder.encoder.weight.zero_()
        autoencoder.encoder.bias.zero_()
        autoencoder.decoder_bias.fill_(1.0)

    reconstructed = reconstruct_trace_site(autoencoder, trace, site=site, layer=0)
    original = trace.layers[0].residual_pre if site == "residual_pre" else trace.layers[0].head_outputs
    if site == "head_output":
        original = original.permute(0, 2, 1, 3).flatten(2)
        reconstructed = reconstructed.permute(0, 2, 1, 3).flatten(2)
    else:
        original = original[:, trace.layout.prefix_length :]
        reconstructed = reconstructed[:, trace.layout.prefix_length :]

    padding = ~trace.layout.sequence_validity
    assert_close(reconstructed[padding], original[padding])


@pytest.mark.cuda
def test_reconstruction_supports_cpu_autoencoder_with_cuda_trace() -> None:
    config = ViTConfig(
        in_channels=1,
        patch_size=(2, 2),
        img_size=(4, 4),
        depth=1,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pos_enc="none",
        dtype=torch.float32,
    )
    model = ViT(config, device=torch.device("cuda")).eval()
    inputs = torch.randn(1, 1, 4, 4, device="cuda")
    explainer = ViTExplainer(model, lambda features: features.visual_tokens.mean((1, 2)))
    trace = explainer.trace(inputs)
    autoencoder = TopKSparseAutoencoder(8, 16, k=4)

    reconstructed = reconstruct_trace_site(autoencoder, trace, site="residual_post", layer=0)

    assert reconstructed.device.type == "cuda"
    assert reconstructed.shape == trace.layers[0].residual_post.shape


def test_sparse_training_restarts_reusable_stream_and_reduces_loss() -> None:
    torch.manual_seed(2)
    autoencoder = TopKSparseAutoencoder(4, 8, k=3)
    batches = [torch.randn(16, 4)]

    losses = train_sparse_autoencoder(autoencoder, batches, steps=30, learning_rate=2e-2)

    assert len(losses) == 30
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]
    assert_close(autoencoder.decoder_directions.norm(dim=1), torch.ones(8))


def test_score_recovery_has_expected_endpoints() -> None:
    original = torch.tensor([3.0])
    ablated = torch.tensor([1.0])
    assert_close(score_recovery(original, original, ablated), torch.ones(1))
    assert_close(score_recovery(original, ablated, ablated), torch.zeros(1))


def test_vit_activation_stream_restores_model_before_yield_and_is_restartable() -> None:
    config = ViTConfig(
        in_channels=1,
        patch_size=(2, 2),
        img_size=(4, 4),
        depth=1,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pos_enc="none",
        dtype=torch.float32,
    )
    model = ViT(config).train()
    explainer = ViTExplainer(model, lambda features: features.visual_tokens.mean((1, 2)))
    dataloader = [(torch.randn(2, 1, 4, 4), ["a", "b"])]
    stream = stream_vit_activations(explainer, dataloader, site="residual_post", layer=0)

    iterator = iter(stream)
    next(iterator)
    assert model.training
    first = list(stream)
    second = list(stream)
    autoencoder = TopKSparseAutoencoder(8, 12, k=3)
    atlas = scan_sparse_features(autoencoder, explainer, dataloader, site="residual_post", layer=0, top_k=1)

    assert_close(first[0], second[0])
    assert model.training
    assert atlas.features
    assert atlas.features[0][0].sample_id in {"a", "b"}


def test_vit_activation_stream_excludes_ragged_padding_tokens() -> None:
    config = ViTConfig(
        in_channels=1,
        patch_size=(2, 2),
        img_size=(4, 4),
        depth=1,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pos_enc="none",
        dtype=torch.float32,
    )
    model = ViT(config).eval()
    inputs = torch.randn(2, 1, 4, 4)
    mask = torch.tensor([[True, True, True, False], [False, True, False, False]])
    explainer = ViTExplainer(model, lambda features: features.visual_tokens.mean((1, 2)))
    arguments = ForwardArgs(mask=mask)
    trace = explainer.trace(inputs, forward_args=arguments)
    expected = trace.layers[0].residual_post[:, trace.layout.prefix_length :][trace.layout.sequence_validity]

    streamed = next(
        iter(
            stream_vit_activations(
                explainer,
                [inputs],
                site="residual_post",
                layer=0,
                forward_args=arguments,
            )
        )
    )

    assert streamed.shape == (int(mask.sum().item()), config.hidden_size)
    assert_close(streamed, expected)


def test_sparse_feature_scan_rejects_incompatible_batch_layouts() -> None:
    config = ViTConfig(
        in_channels=1,
        patch_size=(2, 2),
        img_size=(4, 4),
        depth=1,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pos_enc="none",
        dtype=torch.float32,
    )
    model = ViT(config).eval()
    explainer = ViTExplainer(model, lambda features: features.visual_tokens.mean((1, 2)))
    autoencoder = TopKSparseAutoencoder(8, 12, k=3)
    dataloader = [torch.randn(1, 1, 4, 4), torch.randn(1, 1, 6, 4)]

    with pytest.raises(ValueError, match="layout"):
        scan_sparse_features(autoencoder, explainer, dataloader, site="residual_post", layer=0)


@pytest.mark.cuda
def test_sparse_feature_scan_supports_cpu_autoencoder_with_cuda_model() -> None:
    config = ViTConfig(
        in_channels=1,
        patch_size=(2, 2),
        img_size=(4, 4),
        depth=1,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pos_enc="none",
        dtype=torch.float32,
    )
    model = ViT(config, device=torch.device("cuda")).eval()
    inputs = torch.randn(1, 1, 4, 4, device="cuda")
    explainer = ViTExplainer(model, lambda features: features.visual_tokens.mean((1, 2)))
    autoencoder = TopKSparseAutoencoder(8, 12, k=3)

    atlas = scan_sparse_features(autoencoder, explainer, [inputs], site="residual_post", layer=0, top_k=1)

    assert atlas.features


@pytest.mark.parametrize(
    "arguments",
    [
        {"input_features": 0, "dictionary_features": 4, "k": 1},
        {"input_features": 4, "dictionary_features": 0, "k": 1},
        {"input_features": 4, "dictionary_features": 4, "k": 0},
        {"input_features": 4, "dictionary_features": 4, "k": 5},
    ],
)
def test_sparse_autoencoder_rejects_invalid_dimensions(arguments) -> None:
    with pytest.raises(ValueError):
        TopKSparseAutoencoder(**arguments)
