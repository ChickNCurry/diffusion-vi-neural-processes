from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class DeterministicEncoder(nn.Module):
    def __init__(
        self, x_dim: int, y_dim: int, h_dim: int, r_dim: int, is_attentive: bool = False
    ) -> None:
        super(DeterministicEncoder, self).__init__()

        self.is_attentive = is_attentive

        self.proj_in = nn.Linear(x_dim + y_dim, h_dim)

        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )

        if self.is_attentive:
            self.self_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

        self.proj_out = nn.Linear(h_dim, r_dim)

    def forward(self, x_context: Tensor, y_context: Tensor) -> Tensor:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)

        context = torch.cat([x_context, y_context], dim=-1)
        # -> (batch_size, context_size, x_dim + y_dim)

        context = self.proj_in(context)
        context = self.mlp(context)
        # -> (batch_size, context_size, r_dim)

        if self.is_attentive:
            context = self.self_attn(context, context, context, need_weights=False)
            # -> (batch_size, context_size, r_dim)

        r: Tensor = self.proj_out(context)
        # -> (batch_size, context_size, r_dim)

        if self.is_attentive:
            r = torch.mean(r, dim=1)
            # -> (batch_size, r_dim)

        return r


class LatentEncoder(nn.Module):
    def __init__(
        self, x_dim: int, y_dim: int, h_dim: int, z_dim: int, is_attentive: bool = False
    ) -> None:
        super(LatentEncoder, self).__init__()

        self.is_attentive = is_attentive

        self.proj_in = nn.Linear(x_dim + y_dim, h_dim)

        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )

        if self.is_attentive:
            self.self_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

        self.proj_z_mu = nn.Linear(h_dim, z_dim)
        self.proj_z_w = nn.Linear(h_dim, z_dim)

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
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)

        context = torch.cat([x_context, y_context], dim=-1)
        # -> (batch_size, context_size, x_dim + y_dim)

        context = self.proj_in(context)
        context = self.mlp(context)
        # -> (batch_size, context_size, h_dim)

        if self.is_attentive:
            context = self.self_attn(context, context, context, need_weights=False)
            # -> (batch_size, context_size, h_dim)

        s = torch.mean(context, dim=1)
        # -> (batch_size, h_dim)

        z_mu = self.proj_z_mu(s)
        z_w = self.proj_z_w(s)
        z, z_std = self.reparameterize(z_mu, z_w)
        # -> (batch_size, z_dim)

        return z, z_mu, z_std


class Decoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        r_dim: int,
        z_dim: int,
        h_dim: int,
        y_dim: int,
        is_attentive: bool = False,
    ) -> None:
        super(Decoder, self).__init__()

        self.is_attentive = is_attentive

        self.proj_in = nn.Linear(x_dim + r_dim + z_dim, h_dim)

        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )

        self.proj_y_mu = nn.Linear(h_dim, y_dim)
        self.proj_y_w = nn.Linear(h_dim, y_dim)

    def reparameterize(self, y_mu: Tensor, y_w: Tensor) -> Tuple[Tensor, Tensor]:
        # (batch_size, y_dim)
        # (batch_size, y_dim)

        y_std = 0.1 + 0.9 * nn.Softplus()(y_w)
        # -> (batch_size, y_dim)

        eps = torch.randn_like(y_std)
        # -> (batch_size, y_dim)

        y = y_mu + y_std * eps
        # -> (batch_size, y_dim)

        return y, y_std

    def forward(
        self, x_target: Tensor, r: Tensor, z: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, target_size, x_dim)
        # (batch_size, target_size,r_dim) if is_attentive else (batch_size, r_dim)
        # (batch_size, z_dim)

        if not self.is_attentive:
            r = r.unsqueeze(1).repeat(1, x_target.shape[1], 1)
            # -> (batch_size, target_size, r_dim)

        z = z.unsqueeze(1).repeat(1, x_target.shape[1], 1)
        # -> (batch_size, target_size, z_dim)

        concatted = torch.cat([x_target, r, z], dim=-1)
        # -> (batch_size, target_size, r_dim + z_dim + x_dim)

        concatted = self.proj_in(concatted)
        concatted = self.mlp(concatted)
        # -> (batch_size, target_size, h_dim)

        y_mu = self.proj_y_mu(concatted)
        y_w = self.proj_y_w(concatted)
        y, y_std = self.reparameterize(y_mu, y_w)
        # -> (batch_size, target_size, y_dim)

        return y, y_mu, y_std


class NeuralProcess(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        r_dim: int,
        z_dim: int,
        h_dim: int,
        is_attentive: bool = False,
    ) -> None:
        super(NeuralProcess, self).__init__()

        self.is_attentive = is_attentive

        self.deterministic_encoder = DeterministicEncoder(
            x_dim, y_dim, h_dim, r_dim, is_attentive
        )
        self.latent_encoder = LatentEncoder(x_dim, y_dim, h_dim, z_dim, is_attentive)

        if self.is_attentive:
            self.cross_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

        self.decoder = Decoder(x_dim, r_dim, z_dim, h_dim, y_dim, is_attentive)

    def encode(
        self, x_context: Tensor, y_context: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)

        r = self.deterministic_encoder(x_context, y_context)
        # -> (batch_size, r_dim)

        z, z_mu, z_std = self.latent_encoder(x_context, y_context)
        # -> (batch_size, z_dim)

        return r, z, z_mu, z_std

    def decode(
        self, x_target: Tensor, r: Tensor, z: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, target_size, x_dim)
        # (batch_size, r_dim)
        # (batch_size, z_dim)

        y, y_mu, y_std = self.decoder(x_target, r, z)
        # -> (batch_size, target_size, y_dim)

        return y, y_mu, y_std

    def forward(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)
        # (batch_size, target_size, x_dim)

        r, z, _, _ = self.encode(x_context, y_context)
        # -> (batch_size, context_size, r_dim) if is_attentive else (batch_size, r_dim)
        # -> (batch_size, z_dim)

        if self.is_attentive:
            r = self.cross_attn(x_target, x_context, r, need_weights=False)
            # -> (batch_size, target_size, r_dim)

        y, y_mu, y_std = self.decode(x_target, r, z)
        # -> (batch_size, target_size, y_dim)

        return y, y_mu, y_std
