"""Double Dueling DQN network."""
from __future__ import annotations

import torch
import torch.nn as nn


class DQN(nn.Module):
    """Dueling DQN with a LayerNorm-regularised trunk."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        prev = input_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.advantage_head(h)
        return v + (a - a.mean(dim=1, keepdim=True))
