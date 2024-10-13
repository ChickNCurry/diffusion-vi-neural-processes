from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from deterministic_encoder import DeterministicEncoder
from latent_encoder import LatentEncoder, z_tuple
from torch import Tensor, nn


class DiffusionProcess(nn.Module, ABC):
    def __init__(
        self,
        device: torch.device,
        r_dim: int,
        h_dim: int,
        z_dim: int,
        num_layers: int,
        non_linearity: str,
        num_steps: int,
        det_encoder: DeterministicEncoder,
        lat_encoder: LatentEncoder,
    ) -> None:
        super(DiffusionProcess, self).__init__()

        assert num_steps > 0

        self.num_steps = num_steps
        self.z_dim = z_dim
        self.det_encoder = det_encoder
        self.lat_encoder = lat_encoder

        self.delta_t = torch.tensor(1 / num_steps).to(device)

        self.score_mlp = nn.Sequential(
            nn.Linear(z_dim + r_dim, h_dim),
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 1)
            ],
            nn.Linear(h_dim, z_dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(1, z_dim + r_dim), getattr(nn, non_linearity)()
        )

    @abstractmethod
    def forward_z_mu_and_z_sigma(
        self, z_prev: Tensor, t: int, h: Tensor
    ) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def backward_z_mu_and_z_sigma(
        self, z_next: Tensor, t: int, h: Tensor
    ) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def get_sigma(self, t: int) -> Tensor:
        pass

    def reparameterize(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
        # (batch_size, z_dim)

        eps = torch.randn_like(z_sigma)
        z = z_mu + z_sigma * eps
        # (batch_size, z_dim)

        return z

    def forward(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor
    ) -> List[z_tuple]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)
        # (batch_size, target_size, x_dim)

        z_tuples = self.forward_process(x_context, y_context, x_target)

        return z_tuples

    def forward_process(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor
    ) -> List[z_tuple]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)
        # (batch_size, target_size, x_dim)

        r = self.det_encoder(x_context, y_context, x_target)
        # (batch_size, r_dim)

        z_0_mu = torch.zeros((r.shape[0], self.z_dim), device=r.device)
        z_0_sigma = self.get_sigma(0)
        z_0 = torch.normal(z_0_mu, z_0_sigma)
        # (batch_size, z_dim)

        z_tuples = [z_tuple(z_0, z_0_mu, z_0_sigma)]

        for t in range(1, self.num_steps + 1):

            z_mu, z_sigma = self.forward_z_mu_and_z_sigma(z_tuples[-1].z, t, r)
            z = self.reparameterize(z_mu, z_sigma)

            z_tuples.append(z_tuple(z, z_mu, z_sigma))

        return z_tuples

    def backward_process(
        self,
        z_samples: List[Tensor],
        x_context: Tensor,
        y_context: Tensor,
        x_target: Tensor,
    ) -> List[z_tuple]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)
        # (batch_size, target_size, x_dim)

        r = self.det_encoder(x_context, y_context, x_target)
        # (batch_size, r_dim)

        z_T_tuples: List[z_tuple] = self.lat_encoder(x_context, y_context, x_target)
        # (batch_size, z_dim)

        z_tuples = [z_tuple(z_samples[-1], z_T_tuples[0].z_mu, z_T_tuples[0].z_sigma)]

        for t in range(self.num_steps - 1, 0, -1):

            z_mu, z_sigma = self.backward_z_mu_and_z_sigma(z_samples[t], t, r)

            z_tuples.append(z_tuple(z_samples[t - 1], z_mu, z_sigma))

        return z_tuples


class DIS(DiffusionProcess):
    def __init__(
        self,
        device: torch.device,
        r_dim: int,
        h_dim: int,
        z_dim: int,
        num_layers: int,
        non_linearity: str,
        num_steps: int,
        det_encoder: DeterministicEncoder,
        lat_encoder: LatentEncoder,
    ) -> None:
        super(DIS, self).__init__(
            device,
            r_dim,
            h_dim,
            z_dim,
            num_layers,
            non_linearity,
            num_steps,
            det_encoder,
            lat_encoder,
        )

        self.betas = nn.ParameterList(
            [torch.tensor([t]) for t in torch.linspace(0.001, 10, num_steps).tolist()]
        )

        # self.betas = nn.ParameterList(
        #     [nn.Parameter(torch.tensor(1.0)) for _ in range(num_steps)]
        # )

        self.sigmas = nn.ParameterList([nn.Parameter(torch.ones(z_dim))])

    def forward_z_mu_and_z_sigma(
        self, z_prev: Tensor, t: int, r: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim), (1), (batch_size, r_dim)

        beta_t = self.betas[t - 1]
        # (1)

        time = torch.tensor([t]).float().to(r.device)
        time = self.time_embedding(time).unsqueeze(0)
        time = time.repeat(z_prev.shape[0], 1)
        # (batch_size, z_dim)

        score = torch.cat([z_prev, r], dim=1)
        score = self.score_mlp(score + time)
        # (batch_size, z_dim)

        z_mu = z_prev + (beta_t * z_prev + score) * self.delta_t
        # (batch_size, z_dim)

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigmas[0]
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        return z_mu, z_sigma

    def backward_z_mu_and_z_sigma(
        self, z_next: Tensor, t: int, h: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim), (1), (batch_size, r_dim)

        beta_t = self.betas[t - 1]
        # (1)

        z_mu = (z_next - beta_t * z_next) * self.delta_t
        # (batch_size, z_dim)

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigmas[0]
        z_sigma = z_sigma.repeat(z_mu.shape[0], 1)
        # (batch_size, z_dim)

        return z_mu, z_sigma

    def get_sigma(self, t: int) -> Tensor:
        sigma: Tensor = self.sigmas[0].data
        return sigma
