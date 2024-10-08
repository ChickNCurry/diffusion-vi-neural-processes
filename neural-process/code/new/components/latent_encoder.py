from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from diffusion_process import DiffusionProcess
from torch import Tensor


@dataclass
class z_tuple:
    z: Tensor
    z_mu: Tensor
    z_sigma: Tensor


class LatentEncoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        h_dim: int,
        z_dim: int,
        num_layers: int,
        non_linearity: str,
        is_attentive: bool,
        aggregation: str,
        diffusion_process: Optional[DiffusionProcess],
        fixed_start_density: bool = False,
    ) -> None:
        super(LatentEncoder, self).__init__()

        self.aggregation = getattr(torch, aggregation) if aggregation else None
        self.is_attentive = is_attentive
        self.diffusion_process = diffusion_process
        self.fixed_start_density = fixed_start_density
        self.z_dim = z_dim

        self.mlp = nn.Sequential(
            nn.Linear(x_dim + y_dim, h_dim),
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 1)
            ],
        )

        if self.is_attentive:
            self.self_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

        self.proj_z_mu = nn.Linear(h_dim, z_dim)
        self.proj_z_w = nn.Linear(h_dim, z_dim)
        self.proj_z_r = nn.Linear(h_dim, z_dim)

    def reparameterize(self, z_mu: Tensor, z_w: Tensor) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim)
        # (batch_size, z_dim)

        z_sigma = 0.1 + 0.9 * torch.sigmoid(z_w)
        # -> (batch_size, z_dim)

        eps = torch.randn_like(z_sigma)
        # -> (batch_size, z_dim)

        z = z_mu + z_sigma * eps
        # -> (batch_size, z_dim)

        return z, z_sigma

    def forward(self, x_context: Tensor, y_context: Tensor) -> List[z_tuple]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)

        h = torch.cat([x_context, y_context], dim=-1)
        # -> (batch_size, context_size, x_dim + y_dim)

        h = self.mlp(h)
        # -> (batch_size, context_size, h_dim)

        if self.is_attentive:
            h, _ = self.self_attn(h, h, h, need_weights=False)
            # -> (batch_size, context_size, h_dim)

        h = self.aggregation(h, dim=1)
        # -> (batch_size, h_dim)

        if self.diffusion_process is None or self.fixed_start_density is False:
            z_mu, z_w = self.proj_z_mu(h), self.proj_z_w(h)
            z, z_sigma = self.reparameterize(z_mu, z_w)
            # -> (batch_size, z_dim)

            z_tuples = [z_tuple(z, z_mu, z_sigma)]

        if self.diffusion_process is not None:
            # z_r = self.proj_z_r(h)
            # -> (batch_size, z_dim)

            if self.fixed_start_density is True:
                z_mu = torch.zeros((h.shape[0], self.z_dim), device=h.device)
                z_sigma = self.diffusion_process.sigmas[0]
                z = torch.normal(z_mu, z_sigma).to(h.device)
                # -> (batch_size, z_dim)

                z_tuples = [z_tuple(z, z_mu, z_sigma)]

            for t in range(self.diffusion_process.num_steps):
                z_tuples.append(
                    z_tuple(
                        *self.diffusion_process.forward_transition(z_tuples[-1].z, t, h)
                    )
                )

        return z_tuples
