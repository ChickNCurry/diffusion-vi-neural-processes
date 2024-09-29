from typing import Optional, Tuple

import torch
import torch.nn as nn
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
    ) -> None:
        super(DeterministicEncoder, self).__init__()

        self.is_attentive = is_attentive

        self.proj_in = nn.Linear(x_dim + y_dim, h_dim)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (nn.Linear(h_dim, h_dim), getattr(nn, non_linearity)())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(h_dim, h_dim),
        )

        if self.is_attentive:
            self.self_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

            self.mlp_x_target = nn.Sequential(
                nn.Linear(x_dim, r_dim), nn.SiLU(), nn.Linear(r_dim, r_dim)
            )

            self.mlp_x_context = nn.Sequential(
                nn.Linear(x_dim, r_dim), nn.SiLU(), nn.Linear(r_dim, r_dim)
            )

            self.cross_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

        self.proj_out = nn.Linear(h_dim, r_dim)

    def forward(self, x_context: Tensor, y_context: Tensor, x_target: Tensor) -> Tensor:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)
        # (batch_size, target_size, x_dim)

        h = torch.cat([x_context, y_context], dim=-1)
        # -> (batch_size, context_size, x_dim + y_dim)

        h = self.proj_in(h)
        h = self.mlp(h)
        # -> (batch_size, context_size, h_dim)

        if self.is_attentive:
            h, _ = self.self_attn(h, h, h, need_weights=False)
            # -> (batch_size, context_size, h_dim)

        r: Tensor = self.proj_out(h)
        # -> (batch_size, context_size, r_dim)

        if self.is_attentive:
            x_target = self.mlp_x_target(x_target)
            # -> (batch_size, target_size, r_dim)

            x_context = self.mlp_x_context(x_context)
            # -> (batch_size, context_size, r_dim)

            r, _ = self.cross_attn(x_target, x_context, r, need_weights=False)
            # -> (batch_size, target_size, r_dim)

        else:
            r = torch.mean(r, dim=1)
            # -> (batch_size, r_dim)

            r = r.unsqueeze(1).repeat(1, x_target.shape[1], 1)
            # -> (batch_size, target_size, r_dim)

        return r


class LatentEncoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        h_dim: int,
        z_dim: int,
        num_layers: int,
        non_linearity: str,
    ) -> None:
        super(LatentEncoder, self).__init__()

        self.proj_in = nn.Linear(x_dim + y_dim, h_dim)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (nn.Linear(h_dim, h_dim), getattr(nn, non_linearity)())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(h_dim, h_dim),
        )

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
        # (batch_size, target_size, x_dim)

        s = torch.cat([x_context, y_context], dim=-1)
        # -> (batch_size, context_size, x_dim + y_dim)

        s = self.proj_in(s)
        s = self.mlp(s)
        # -> (batch_size, context_size, h_dim)

        s = torch.mean(s, dim=1)
        # -> (batch_size, h_dim)

        z_mu, z_w = self.proj_z_mu(s), self.proj_z_w(s)
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
        num_layers: int,
        non_linearity: str,
        has_latent_path: bool,
        has_deterministic_path: bool,
    ) -> None:
        super(Decoder, self).__init__()

        self.has_latent_path = has_latent_path
        self.has_deterministic_path = has_deterministic_path

        match self.has_latent_path, self.has_deterministic_path:
            case True, True:
                self.proj_in = nn.Linear(x_dim + r_dim + z_dim, h_dim)
            case True, False:
                self.proj_in = nn.Linear(x_dim + z_dim, h_dim)
            case False, True:
                self.proj_in = nn.Linear(x_dim + r_dim, h_dim)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (nn.Linear(h_dim, h_dim), getattr(nn, non_linearity)())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(h_dim, h_dim),
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

    def forward(
        self, x_target: Tensor, r: Optional[Tensor], z: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, target_size, x_dim)
        # (batch_size, target_size, r_dim)
        # (batch_size, z_dim)

        if z is not None:
            z = z.unsqueeze(1).repeat(1, x_target.shape[1], 1)
            # -> (batch_size, target_size, z_dim)

        match self.has_latent_path, self.has_deterministic_path:
            case True, True:
                if r is not None and z is not None:
                    h = torch.cat([x_target, r, z], dim=-1)
                    # -> (batch_size, target_size, r_dim + z_dim + x_dim)
            case True, False:
                if z is not None:
                    h = torch.cat([x_target, z], dim=-1)
                    # -> (batch_size, target_size, z_dim + x_dim)
            case False, True:
                if r is not None:
                    h = torch.cat([x_target, r], dim=-1)
                    # -> (batch_size, target_size, r_dim + x_dim)

        h = self.proj_in(h)
        h = self.mlp(h)
        # -> (batch_size, target_size, h_dim)

        y_mu, y_w = self.proj_y_mu(h), self.proj_y_w(h)
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
        num_layers_det_enc: int,
        num_layers_lat_enc: int,
        num_layers_dec: int,
        non_linearity: str,
        is_attentive: bool,
        has_latent_path: bool,
        has_deterministic_path: bool,
    ) -> None:
        super(NeuralProcess, self).__init__()

        self.has_latent_path = has_latent_path
        self.has_deterministic_path = has_deterministic_path

        if self.has_deterministic_path:
            self.deterministic_encoder = DeterministicEncoder(
                x_dim,
                y_dim,
                h_dim,
                r_dim,
                num_layers_det_enc,
                non_linearity,
                is_attentive,
            )

        if self.has_latent_path:
            self.latent_encoder = LatentEncoder(
                x_dim,
                y_dim,
                h_dim,
                z_dim,
                num_layers_lat_enc,
                non_linearity,
            )

        self.decoder = Decoder(
            x_dim,
            r_dim,
            z_dim,
            h_dim,
            y_dim,
            num_layers_dec,
            non_linearity,
            has_latent_path,
            has_deterministic_path,
        )

    def forward(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)
        # (batch_size, target_size, x_dim)

        z, z_mu, z_std = (
            self.latent_encoder(x_context, y_context)
            if self.has_latent_path
            else (None, None, None)
        )
        # -> (batch_size, target_size, z_dim)

        r = (
            self.deterministic_encoder(x_context, y_context, x_target)
            if self.has_deterministic_path
            else None
        )
        # -> (batch_size, target_size, r_dim)

        y, y_mu, y_std = self.decoder(x_target, r, z)
        # -> (batch_size, target_size, y_dim)

        return y, y_mu, y_std, z, z_mu, z_std

    def sample(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor, num_samples: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # (1, context_size, x_dim)
        # (1, context_size, y_dim)
        # (1, target_size, x_dim)

        x_context = x_context.repeat(num_samples, 1, 1)
        # -> (num_samples, context_size, x_dim)

        y_context = y_context.repeat(num_samples, 1, 1)
        # -> (num_samples, context_size, y_dim)

        x_target = x_target.repeat(num_samples, 1, 1)
        # -> (num_samples, target_size, x_dim)

        y, y_mu, y_std, z, z_mu, z_std = self.forward(x_context, y_context, x_target)
        # -> (num_samples, target_size, y_dim), (num_samples, z_dim)

        return y, y_mu, y_std, z, z_mu, z_std
