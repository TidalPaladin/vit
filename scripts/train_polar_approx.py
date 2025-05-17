import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from tqdm import tqdm

from vit.attention import PolarApprox


def symmetric_linear_falloff(r: Tensor, theta: Tensor) -> Tensor:
    return r


def symmetric_quadratic_falloff(r: Tensor, theta: Tensor) -> Tensor:
    return r.pow(2)


def top_left_linear_falloff(r: Tensor, theta: Tensor) -> Tensor:
    angle_diff = torch.abs(theta - 3 * torch.pi / 4)
    angle_factor = 2.0 * angle_diff / torch.pi
    return r * angle_factor


def left_right_linear_falloff(r: Tensor, theta: Tensor) -> Tensor:
    return r * theta.sin().pow(2)


def top_bottom_linear_falloff(r: Tensor, theta: Tensor) -> Tensor:
    return r * theta.cos().pow(2)


def cross(r: Tensor, theta: Tensor) -> Tensor:
    return r * theta.mul(2).cos().pow(2)


def plus(r: Tensor, theta: Tensor) -> Tensor:
    return r * theta.add(torch.pi / 4).mul(2).cos().pow(2)


def well(r: Tensor, theta: Tensor, threshold: float = 0.5) -> Tensor:
    return torch.where(r < threshold, 0.0, 1.0)


FUNCTIONS = {
    "symmetric": symmetric_linear_falloff,
    "symmetric_quadratic": symmetric_quadratic_falloff,
    "top_left": top_left_linear_falloff,
    "left_right": left_right_linear_falloff,
    "top_bottom": top_bottom_linear_falloff,
    "cross": cross,
    "plus": plus,
    "well": well,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-r", "--radial-degree", type=int, default=2)
    parser.add_argument("-a", "--angular-degree", type=int, default=4)
    parser.add_argument("-b", "--batch-size", type=int, default=1024)
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("-n", "--num-steps", type=int, default=100)
    parser.add_argument("-d", "--device", type=torch.device, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-o", "--output", type=Path, default=os.getcwd())
    parser.add_argument("-s", "--scale", type=float, default=10)
    parser.add_argument("-i", "--initial", default=False, action="store_true")
    parser.add_argument(
        "-f",
        "--function",
        choices=FUNCTIONS.keys(),
        default="symmetric",
    )
    return parser.parse_args()


def main(args: Namespace):
    layer = PolarApprox(args.radial_degree, args.angular_degree).to(args.device)
    optim = AdamW(layer.parameters(), lr=args.learning_rate, weight_decay=0.01)
    func = FUNCTIONS[args.function]
    if not args.initial:
        for _ in tqdm(range(args.num_steps)):
            r = torch.rand(args.batch_size, 1, device=args.device) * args.scale
            theta = torch.rand(args.batch_size, 1, device=args.device) * 2 * torch.pi
            y = func(r, theta)
            with torch.autocast(device_type=args.device.type, dtype=torch.bfloat16):
                pred = layer(r, theta)
                loss = F.mse_loss(pred.flatten(), y.flatten())
            optim.zero_grad()
            loss.backward()
            optim.step()

    layer.plot(
        r_min=0,
        r_max=args.scale,
        title=f"Angular Degree: {args.angular_degree}, Radial Degree: {args.radial_degree}",
        filename=args.output / f"{args.function}.png",
    )


if __name__ == "__main__":
    main(parse_args())
