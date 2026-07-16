"""Create, inspect, evaluate, and save a synthetic ViT explanation."""

from pathlib import Path

import torch

from vit import AttentivePoolHeadConfig, ViTConfig
from vit.explain import DeletionInsertion, LeGrad, ViTExplainer, save_explanation


def main() -> None:
    torch.manual_seed(4)
    config = ViTConfig(
        in_channels=1,
        patch_size=(4, 4),
        img_size=(16, 16),
        depth=2,
        hidden_size=16,
        ffn_hidden_size=32,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pos_enc="rope",
        dtype=torch.float32,
        heads={"prediction": AttentivePoolHeadConfig(out_features=3)},
    )
    model = config.instantiate().eval()
    with torch.no_grad():
        for layer in range(model.config.depth):
            block = model.get_block(layer)
            torch.nn.init.xavier_uniform_(block.self_attention.out_proj.weight)
            torch.nn.init.xavier_uniform_(block.mlp.fc2.weight)

    inputs = torch.randn(1, 1, 16, 16)
    explainer = ViTExplainer.from_head(model, "prediction")
    explanation = explainer.attribute(inputs, target=1, method=LeGrad())
    report = explainer.evaluate(
        inputs,
        explanation,
        target=1,
        metrics=[DeletionInsertion(steps=4)],
    )
    output = Path("synthetic-explanation.npz")
    save_explanation(explanation, output, overwrite=True)
    print(f"saved {output}")
    print(f"token attribution shape: {tuple(explanation.token_attributions.shape)}")
    print(f"evaluation metrics: {', '.join(report.metrics)}")


if __name__ == "__main__":
    main()
