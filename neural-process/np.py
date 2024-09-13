from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor

# context_dim for sequence is 2 (x, v) -> x_dim 1 & y_dim 1
# context_dim for grey img is 3 (x, y, v) -> x_dim 2 & y_dim 1
# context_dim for color img is 2 (c, x, y, v) -> x_dim 3 & y_dim 1


class SetEncoder(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, h_dim: int, r_dim: int) -> None:
        super(SetEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, r_dim),
        )

    def forward(self, c: Tensor) -> Tensor:
        # (batch_size, context_size, context_dim)

        c = self.encoder(c)
        # -> (batch_size, context_size, r_dim)

        r = torch.mean(c, dim=1)
        # -> (batch_size, r_dim)

        return r


class LatentEncoder(nn.Module):
    def __init__(self, r_dim: int, z_dim: int) -> None:
        super(LatentEncoder, self).__init__()

        self.proj_mu = nn.Linear(r_dim, z_dim)
        self.proj_logvar = nn.Linear(r_dim, z_dim)

    def latent_encode(self, r: Tensor) -> Tuple[Tensor, Tensor]:
        # (batch_size, r_dim)

        mu, logvar = self.proj_mu(r), self.proj_logvar(r)
        # -> (batch_size, z_dim)

        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # (batch_size, z_dim), (batch_size, z_dim)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        # -> (batch_size, z_dim)

        return z

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, r_dim)

        mu, logvar = self.latent_encode(x)
        # -> (batch_size, z_dim)

        z = self.reparameterize(mu, logvar)
        # -> (batch_size, z_dim)

        return z, mu, logvar

    def sample(self, x: Tensor, n_samples: int) -> Tuple[List[Tensor], Tensor, Tensor]:
        # (batch_size, r_dim)

        mu, logvar = self.latent_encode(x)
        # -> (batch_size, z_dim)

        zs = [self.reparameterize(mu, logvar) for _ in range(n_samples)]
        # -> (batch_size, z_dim)

        return zs, mu, logvar


class Decoder(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, z_dim: int, h_dim: int) -> None:
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim + x_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )

        self.proj_mu = nn.Linear(h_dim, y_dim)
        self.proj_logvar = nn.Linear(h_dim, y_dim)

    def forward(self, z: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim), (batch_size, target_size, x_dim)

        z = z.unsqueeze(1).repeat(1, t.shape[1], 1)
        # -> (batch_size, target_size, z_dim)

        z_t = torch.cat([z, t], dim=2)
        # -> (batch_size, target_size, z_dim + x_dim)

        logits = self.decoder(z_t)
        # -> (batch_size, target_size, h_dim)

        mu, logvar = self.proj_mu(logits), self.proj_logvar(logits)
        # -> (batch_size, target_size, y_dim)

        return mu, logvar


class NeuralProcess(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        r_dim: int,
        z_dim: int,
        h_dim: int,
    ) -> None:
        super(NeuralProcess, self).__init__()

        self.set_encoder = SetEncoder(x_dim, y_dim, h_dim, r_dim)
        self.latent_encoder = LatentEncoder(r_dim, z_dim)
        self.decoder = Decoder(x_dim, y_dim, z_dim, h_dim)

    def encode(self, c: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, context_size, context_dim)

        r = self.set_encoder(c)
        # -> (batch_size, r_dim)

        z, z_mu, z_logvar = self.latent_encoder(r)
        # -> (batch_size, z_dim)

        return z, z_mu, z_logvar

    def decode(self, z: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim), (batch_size, target_size, x_dim)

        mu, logvar = self.decoder(z, t)
        # -> (batch_size, target_size, y_dim)

        return mu, logvar

    def forward(self, c: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # (batch_size, context_size, context_dim), (batch_size, target_size, x_dim)

        r = self.set_encoder(c)
        # -> (batch_size, r_dim)

        z, z_mu, z_logvar = self.latent_encoder(r)
        # -> (batch_size, z_dim)

        mu, logvar = self.decoder(z, t)
        # -> (batch_size, target_size, y_dim)

        return mu, logvar, z_mu, z_logvar

    def sample(
        self, c: Tensor, t: Tensor, n_samples: int
    ) -> Tuple[Tuple[Tensor], Tuple[Tensor], Tensor, Tensor]:
        # (batch_size, context_size, context_dim), (batch_size, target_size, x_dim)

        r = self.set_encoder(c)
        # -> (batch_size, r_dim)

        zs, z_mu, z_logvar = self.latent_encoder.sample(r, n_samples)
        # -> (batch_size, z_dim)

        out = [self.decoder(z, t) for z in zs]
        mus, logvars = zip(*out)
        # -> (batch_size, target_size, y_dim)

        return mus, logvars, z_mu, z_logvar


if __name__ == "__main__":

    c = torch.randn(5, int(0.7 * 28 * 28), 3)
    t = torch.randn(5, int(0.3 * 28 * 28), 2)

    print(c.shape)
    print(t.shape)

    np = NeuralProcess(x_dim=2, y_dim=1, r_dim=8, z_dim=8, h_dim=32)

    mu, logvar, z_mu, z_logvar = np(c, t)

    print(mu.shape)
    print(logvar.shape)
    print(z_mu.shape)
    print(z_logvar.shape)
