from argparse import ArgumentParser, Namespace
from copy import deepcopy
from time import time

import numpy as np
import torch
from torchao.quantization import Int8DynamicActivationInt8WeightConfig

from vit import ViT, ViTConfig


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--ffn-hidden-size", type=int, default=3072)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--activation", type=str, default="swiglu")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    return parser.parse_args()


def benchmark(model: ViT, batch_size: int, device: torch.device, num_runs: int = 10) -> list[float]:
    C = model.config.in_channels
    H, W = model.config.img_size
    x = torch.randn(batch_size, C, H, W, device=device)
    times = []
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
        # Warmup
        for _ in range(3):
            model(x)
            torch.cuda.synchronize()

        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time()
            model(x)
            torch.cuda.synchronize()
            end_time = time()
            times.append(end_time - start_time)
    return times


def main(args: Namespace):
    config = ViTConfig(
        in_channels=args.in_channels,
        patch_size=(args.patch_size, args.patch_size),
        img_size=(args.img_size, args.img_size),
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        depth=args.depth,
        activation=args.activation,
    )
    model = ViT(config).to(torch.device(args.device))
    model.eval()
    quantized_model = deepcopy(model)

    quantization_config = Int8DynamicActivationInt8WeightConfig()
    quantized_model.apply_quantization(
        mlp_quantization_config=quantization_config,
        qkv_quantization_config=quantization_config,
        attn_quantization_config=quantization_config,
    )

    times_baseline = benchmark(model, args.batch_size, torch.device(args.device))
    times_quantized = benchmark(quantized_model, args.batch_size, torch.device(args.device))

    print(f"Baseline time: {np.mean(times_baseline)} ± {np.std(times_baseline)}")
    print(f"Quantized time: {np.mean(times_quantized)} ± {np.std(times_quantized)}")


def entrypoint():
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
