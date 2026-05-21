from __future__ import annotations

import torch
from torch import nn


class ElementwiseFCN(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 3,
        hidden: tuple[int, ...] = (128, 128, 128),
        activation: str = "silu",
    ) -> None:
        super().__init__()
        activations = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        width_in = input_dim
        for width_out in hidden:
            layers.append(nn.Linear(width_in, width_out))
            layers.append(activations[activation]())
            width_in = width_out
        layers.append(nn.Linear(width_in, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

