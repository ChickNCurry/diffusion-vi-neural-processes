from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from aggregation import Aggregation
from diffusion_process import DiffusionProcess
from torch import Tensor


class LatentEncoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        s_dim: int,
        z_dim: int,
        num_layers: int,
        non_linearity: str,
        is_attentive: bool,
        aggregation: Aggregation,
        diffusion_process: Optional[DiffusionProcess],
    ) -> None:
        super(LatentEncoder, self).__init__()

        self.aggregation = aggregation
        self.is_attentive = is_attentive
        self.diffusion_process = diffusion_process

        self.mlp = nn.Sequential(
            nn.Linear(x_dim + y_dim, s_dim),
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(s_dim, s_dim))
                for _ in range(num_layers - 1)
            ],
        )

        if self.is_attentive:
            self.self_attn = nn.MultiheadAttention(s_dim, 1, batch_first=True)

        self.proj_z_mu = nn.Linear(s_dim, z_dim)
        self.proj_z_w = nn.Linear(s_dim, z_dim)

    def reparameterize(self, z_mu: Tensor, z_w: Tensor) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim)
        # (batch_size, z_dim)

        z_std = 0.1 + 0.9 * torch.sigmoid(z_w)
        # -> (batch_size, z_dim)

        eps = torch.randn_like(z_std)
        # -> (batch_size, z_dim)

        z = z_mu + z_std * eps
        # -> (batch_size, z_dim)

        return z, z_std

    def forward(
        self, x_context: Tensor, y_context: Tensor
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)

        s = torch.cat([x_context, y_context], dim=-1)
        # -> (batch_size, context_size, x_dim + y_dim)

        s = self.mlp(s)
        # -> (batch_size, context_size, h_dim)

        if self.is_attentive:
            s, _ = self.self_attn(s, s, s, need_weights=False)
            # -> (batch_size, context_size, h_dim)

        s = self.aggregation.value(s, dim=1)
        # -> (batch_size, h_dim)

        z_mu, z_w = self.proj_z_mu(s), self.proj_z_w(s)
        z, z_std = self.reparameterize(z_mu, z_w)
        # -> (batch_size, z_dim)

        z_tuples = [(z, z_mu, z_std)]

        if self.diffusion_process is not None:
            for t in range(self.diffusion_process.num_steps):
                z_tuples.append(
                    self.diffusion_process.forward_transition(
                        z_tuples[-1][0], torch.tensor([t]).to(s.device), s
                    )
                )

        return z_tuples
