from typing import Optional

import torch
import torch.nn as nn
from aggregation import Aggregation
from torch import Tensor


class DeterministicEncoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        h_dim: int,
        r_dim: int,
        num_layers: int,
        non_linearity: str,
        is_attentive: bool,
        aggregation: Optional[Aggregation],
    ) -> None:
        super(DeterministicEncoder, self).__init__()

        self.is_attentive = is_attentive
        self.aggregation = aggregation

        assert self.is_attentive == (self.aggregation is None)

        self.mlp = nn.Sequential(
            nn.Linear(x_dim + y_dim, h_dim),
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 1)
            ]
        )

        if self.is_attentive:
            self.self_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

            self.mlp_x_target = nn.Sequential(
                nn.Linear(x_dim, r_dim),
                getattr(nn, non_linearity)(),
                nn.Linear(r_dim, r_dim),
            )

            self.mlp_x_context = nn.Sequential(
                nn.Linear(x_dim, r_dim),
                getattr(nn, non_linearity)(),
                nn.Linear(r_dim, r_dim),
            )

            self.cross_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

        self.proj_r = nn.Linear(h_dim, r_dim)

    def forward(self, x_context: Tensor, y_context: Tensor, x_target: Tensor) -> Tensor:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)
        # (batch_size, target_size, x_dim)

        h = torch.cat([x_context, y_context], dim=-1)
        # -> (batch_size, context_size, x_dim + y_dim)

        h = self.mlp(h)
        # -> (batch_size, context_size, h_dim)

        if self.is_attentive:
            h, _ = self.self_attn(h, h, h, need_weights=False)
            # -> (batch_size, context_size, h_dim)

        r: Tensor = self.proj_r(h)
        # -> (batch_size, context_size, r_dim)

        if self.is_attentive:
            x_target = self.mlp_x_target(x_target)
            # -> (batch_size, target_size, r_dim)

            x_context = self.mlp_x_context(x_context)
            # -> (batch_size, context_size, r_dim)

            r, _ = self.cross_attn(x_target, x_context, r, need_weights=False)
            # -> (batch_size, target_size, r_dim)

        elif self.aggregation is not None:
            r = self.aggregation.value(r, dim=1)
            # -> (batch_size, r_dim)

            r = r.unsqueeze(1).repeat(1, x_target.shape[1], 1)
            # -> (batch_size, target_size, r_dim)

        return r
