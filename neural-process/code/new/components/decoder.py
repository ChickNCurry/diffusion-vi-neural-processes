from typing import Tuple

import torch
from torch import Tensor, nn


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        h_dim: int,
        y_dim: int,
        num_layers: int,
        non_linearity: str,
    ) -> None:
        super(Decoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 1)
            ]
        )

        self.proj_y_mu = nn.Linear(h_dim, y_dim)
        self.proj_y_w = nn.Linear(h_dim, y_dim)

    def reparameterize(self, y_mu: Tensor, y_w: Tensor) -> Tuple[Tensor, Tensor]:
        # (batch_size, target_size, y_dim)
        # (batch_size, target_size, y_dim)

        y_std = 0.1 + 0.9 * nn.Softplus()(y_w)
        # -> (batch_size, target_size, y_dim)

        eps = torch.randn_like(y_std)
        # -> (batch_size, target_size, y_dim)

        y = y_mu + y_std * eps
        # -> (batch_size, target_size, y_dim)

        return y, y_std

    def forward(self, x_target: Tensor, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, target_size, x_dim)
        # (batch_size, target_size, input_dim)

        h = torch.cat([x_target, input], dim=-1)
        # -> (batch_size, target_size, input_dim + x_dim)

        h = self.mlp(h)
        # -> (batch_size, target_size, h_dim)

        y_mu, y_w = self.proj_y_mu(h), self.proj_y_w(h)
        y, y_std = self.reparameterize(y_mu, y_w)
        # -> (batch_size, target_size, y_dim)

        return y, y_mu, y_std
