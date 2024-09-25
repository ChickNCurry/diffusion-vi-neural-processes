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
            nn.Linear(h_dim, h_dim)
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

        context = torch.cat([x_context, y_context], dim=-1)
        # -> (batch_size, context_size, x_dim + y_dim)

        context = self.proj_in(context)
        context = self.mlp(context)
        # -> (batch_size, context_size, h_dim)

        if self.is_attentive:
            context, _ = self.self_attn(context, context, context, need_weights=False)
            # -> (batch_size, context_size, h_dim)

        r: Tensor = self.proj_out(context)
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
        is_attentive: bool,
    ) -> None:
        super(LatentEncoder, self).__init__()

        self.is_attentive = is_attentive

        self.proj_in = nn.Linear(x_dim + y_dim, h_dim)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (nn.Linear(h_dim, h_dim), getattr(nn, non_linearity)())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(h_dim, h_dim)
        )

        if self.is_attentive:
            self.self_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

            self.mlp_x_target = nn.Sequential(
                nn.Linear(x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
            )

            self.mlp_x_context = nn.Sequential(
                nn.Linear(x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
            )

            self.cross_attn = nn.MultiheadAttention(h_dim, 1, batch_first=True)

        self.proj_z_mu = nn.Linear(h_dim, z_dim)
        self.proj_z_w = nn.Linear(h_dim, z_dim)

    def reparameterize(self, z_mu: Tensor, z_w: Tensor) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim)
        # (batch_size, z_dim)

        z_std = nn.Softplus()(z_w)  # 0.1 + 0.9 * torch.sigmoid(z_w)
        # -> (batch_size, z_dim)

        eps = torch.randn_like(z_std)
        # -> (batch_size, z_dim)

        z = z_mu + z_std * eps
        # -> (batch_size, z_dim)

        return z, z_std

    def get_z_mu_and_z_w(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)

        s = torch.cat([x_context, y_context], dim=-1)
        # -> (batch_size, context_size, x_dim + y_dim)

        s = self.proj_in(s)
        s = self.mlp(s)
        # -> (batch_size, context_size, h_dim)

        if self.is_attentive:
            s, _ = self.self_attn(s, s, s, need_weights=False)
            # -> (batch_size, context_size, h_dim)

            x_target = self.mlp_x_target(x_target)
            # -> (batch_size, target_size, h_dim)

            x_context = self.mlp_x_context(x_context)
            # -> (batch_size, context_size, h_dim)

            s, _ = self.cross_attn(x_target, x_context, s, need_weights=False)
            # -> (batch_size, target_size, h_dim)

        else:
            s = torch.mean(s, dim=1)
            # -> (batch_size, h_dim)

            s = s.unsqueeze(1).repeat(1, x_target.shape[1], 1)
            # -> (batch_size, target_size, h_dim)

        z_mu = self.proj_z_mu(s)
        z_w = self.proj_z_w(s)
        # -> (batch_size, target_size, z_dim)

        return z_mu, z_w

    def forward(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)
        # (batch_size, target_size, x_dim)

        z_mu, z_w = self.get_z_mu_and_z_w(x_context, y_context, x_target)
        z, z_std = self.reparameterize(z_mu, z_w)
        # -> (batch_size, target_size, z_dim)

        return z, z_mu, z_std

    def sample(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor, num_samples: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (1, context_size, x_dim)
        # (1, context_size, y_dim)
        # (1, target_size, x_dim)

        z_mu, z_w = self.get_z_mu_and_z_w(x_context, y_context, x_target)
        # -> (1, target_size, z_dim)

        z_samples = []
        z_std_samples = []

        for _ in range(num_samples):
            z, z_std = self.reparameterize(z_mu, z_w)
            # -> (1, target_size, z_dim)

            z_samples.append(z)
            z_std_samples.append(z_std)

        z, z_std = torch.cat(z_samples, dim=0), torch.cat(z_std_samples, dim=0)
        # -> (num_samples, target_size, z_dim)

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
        has_parallel_paths: bool,
    ) -> None:
        super(Decoder, self).__init__()

        self.has_parallel_paths = has_parallel_paths

        if self.has_parallel_paths:
            self.proj_in = nn.Linear(x_dim + r_dim + z_dim, h_dim)
        else:
            self.proj_in = nn.Linear(x_dim + z_dim, h_dim)

        self.mlp = nn.Sequential(
            *[
                layer
                for layer in (nn.Linear(h_dim, h_dim), getattr(nn, non_linearity)())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(h_dim, h_dim)
        )

        self.proj_y_mu = nn.Linear(h_dim, y_dim)
        self.proj_y_w = nn.Linear(h_dim, y_dim)

    def reparameterize(self, y_mu: Tensor, y_w: Tensor) -> Tuple[Tensor, Tensor]:
        # (batch_size, y_dim)
        # (batch_size, y_dim)

        y_std = nn.Softplus()(y_w)  # 0.1 + 0.9 * nn.Softplus()(y_w)
        # -> (batch_size, y_dim)

        eps = torch.randn_like(y_std)
        # -> (batch_size, y_dim)

        y = y_mu + y_std * eps
        # -> (batch_size, y_dim)

        return y, y_std

    def forward(
        self, x_target: Tensor, r: Optional[Tensor], z: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, target_size, x_dim)
        # (batch_size, target_size, z_dim)
        # (batch_size, target_size, r_dim)

        if self.has_parallel_paths and r is not None:
            concatted = torch.cat([x_target, r, z], dim=-1)
            # -> (batch_size, target_size, r_dim + z_dim + x_dim)
        else:
            concatted = torch.cat([x_target, z], dim=-1)
            # -> (batch_size, target_size, z_dim + x_dim)

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
        num_layers_det_enc: int,
        num_layers_lat_enc: int,
        num_layers_dec: int,
        non_linearity: str,
        is_attentive: bool,
        has_parallel_paths: bool,
    ) -> None:
        super(NeuralProcess, self).__init__()

        self.has_parallel_paths = has_parallel_paths

        if self.has_parallel_paths:
            self.deterministic_encoder = DeterministicEncoder(
                x_dim,
                y_dim,
                h_dim,
                r_dim,
                num_layers_det_enc,
                non_linearity,
                is_attentive,
            )

        self.latent_encoder = LatentEncoder(
            x_dim, y_dim, h_dim, z_dim, num_layers_lat_enc, non_linearity, is_attentive
        )

        self.decoder = Decoder(
            x_dim,
            r_dim,
            z_dim,
            h_dim,
            y_dim,
            num_layers_dec,
            non_linearity,
            has_parallel_paths,
        )

    def encode(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor
    ) -> Tuple[Optional[Tensor], Tensor, Tensor, Tensor]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)

        z, z_mu, z_std = self.latent_encoder(x_context, y_context, x_target)
        # -> (batch_size, target_size, z_dim)

        if self.has_parallel_paths:
            r = self.deterministic_encoder(x_context, y_context, x_target)
            # -> (batch_size, target_size, r_dim)
        else:
            r = None

        return r, z, z_mu, z_std

    def decode(
        self, x_target: Tensor, r: Optional[Tensor], z: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, target_size, x_dim)
        # (batch_size, target_size, r_dim)
        # (batch_size, target_size, z_dim)

        y, y_mu, y_std = self.decoder(x_target, r, z)
        # -> (batch_size, target_size, y_dim)

        return y, y_mu, y_std

    def forward(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)
        # (batch_size, target_size, x_dim)

        r, z, _, _ = self.encode(x_context, y_context, x_target)
        # -> (batch_size, target_size, r_dim)
        # -> (batch_size, target_size, z_dim)

        y, y_mu, y_std = self.decode(x_target, r, z)
        # -> (batch_size, target_size, y_dim)

        return y, y_mu, y_std

    def sample(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor, num_samples: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # (1, context_size, x_dim)
        # (1, context_size, y_dim)
        # (1, target_size, x_dim)

        if self.has_parallel_paths:
            r = self.deterministic_encoder(x_context, y_context, x_target)
            # -> (1, target_size, r_dim)

            r = r.repeat(num_samples, 1, 1)
            # -> (num_samples, target_size, r_dim)
        else:
            r = None

        z, z_mu, z_std = self.latent_encoder.sample(
            x_context, y_context, x_target, num_samples
        )
        # -> (num_samples, target_size, z_dim)

        x_target = x_target.repeat(num_samples, 1, 1)
        # -> (num_samples, target_size, x_dim)

        y, y_mu, y_std = self.decoder(x_target, r, z)
        # -> (num_samples, target_size, y_dim)

        return y, y_mu, y_std, z, z_mu, z_std
