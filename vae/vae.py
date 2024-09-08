from typing import Tuple

import torch
from torch import Tensor, nn


class VAE(nn.Module):

    def __init__(self, input_dim: int, h_dim: int = 200, z_dim: int = 20) -> None:
        super(VAE, self).__init__()

        # encoder
        self.fc_in = nn.Linear(input_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

        # decoder
        self.fc_z = nn.Linear(z_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, input_dim)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.relu(self.fc_in(x))
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        z = torch.relu(self.fc_z(z))
        logits: Tensor = self.fc_out(z)
        return logits

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def sample(self, mu: Tensor, logvar: Tensor, num_samples: int) -> Tensor:
        mu = mu.unsqueeze(0).repeat(num_samples, 1, 1)
        logvar = logvar.unsqueeze(0).repeat(num_samples, 1, 1)

        z = self.reparameterize(mu, logvar)

        # sigmoid rlly important because we are using BCEWithLogitsLoss for training
        return torch.sigmoid(self.decode(z))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == "__main__":
    x = torch.randn(4, 28 * 28)
    vae = VAE(28 * 28)
    out, mu, sigma = vae(x)

    print(out.shape)
    print(mu.shape)
    print(sigma.shape)
