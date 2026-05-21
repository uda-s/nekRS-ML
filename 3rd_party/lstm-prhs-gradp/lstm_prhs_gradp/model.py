from __future__ import annotations

import torch
from torch import nn


class ElementwiseLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 3,
        hidden_channels: int = 64,
        n_lstm_layers: int = 2,
        n_mlp_hidden_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_channels,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        layers: list[nn.Module] = []
        width = hidden_channels
        for _ in range(n_mlp_hidden_layers):
            layers.append(nn.Linear(width, hidden_channels))
            layers.append(nn.SiLU())
            width = hidden_channels
        layers.append(nn.Linear(width, output_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

