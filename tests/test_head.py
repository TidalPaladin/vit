import torch.nn as nn

from vit.head import HeadConfig
from vit.vit import ViTConfig


class TestHeadConfig:

    def test_instantiate(self):
        config = HeadConfig(key="[CLS]", out_dim=128, stop_gradient=False)
        vit_config = ViTConfig(
            in_channels=3,
            patch_size=(16, 16),
            img_size=(224, 224),
            depth=3,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=128 // 16,
        )
        model = config.instantiate(vit_config)
        assert isinstance(model, nn.Linear)
