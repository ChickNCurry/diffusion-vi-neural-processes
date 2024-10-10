from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import Tensor, nn


class DiffusionProcess(nn.Module, ABC):
    def __init__(self, device: torch.device, num_steps: int) -> None:
        super(DiffusionProcess, self).__init__()

        self.num_steps = num_steps
        self.delta_t = (
            torch.tensor(1.0 / num_steps).to(device) if num_steps > 0 else None
        )
        self.sigmas: nn.ParameterList

    # def forward(self, z_0: Tensor, h: Tensor) -> Tuple[List[Tensor], Tensor]:
    #     # (batch_size, z_dim), (batch_size, h_dim)

    #     f_log_likes = []
    #     b_log_likes = []

    #     z_list = [z_0]

    #     for t in range(1, self.num_steps + 1):
    #         z_next, z_mu_next, z_sigma_next = self.forward_transition(z_list[-1], t, h)
    #         z_mu, z_sigma = self.backward_transition(z_next, t - 1, h)

    #         f_log_likes.append(Normal(z_mu_next, z_sigma_next).log_prob(z_next))  # type: ignore
    #         b_log_likes.append(Normal(z_mu, z_sigma).log_prob(z_list[-1]))  # type: ignore

    #         z_list.append(z_next)

    #     log_like = torch.stack(b_log_likes).sum() - torch.stack(f_log_likes).sum()

    #     return z_list, log_like

    def reparameterize(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
        # (batch_size, z_dim),
        # (batch_size, z_dim)

        eps = torch.randn_like(z_sigma)
        # (batch_size, z_dim)

        z = z_mu + z_sigma * eps
        # (batch_size, z_dim)

        return z

    def forward_transition(
        self, z_prev: Tensor, t: int, h: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, z_dim), (1), (batch_size, context_size, x_dim)

        z_mu, z_sigma = self.forward_z_mu_and_z_sigma(z_prev, t, h)
        # (batch_size, z_dim), (batch_size, z_dim)

        z = self.reparameterize(z_mu, z_sigma)
        # (batch_size, z_dim)

        return z, z_mu, z_sigma

    def backward_transition(
        self, z_next: Tensor, t: int, h: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim), (1), (batch_size, context_size, x_dim)

        z_mu, z_sigma = self.backward_z_mu_and_z_sigma(z_next, t, h)
        # (batch_size, z_dim), (batch_size, z_dim)

        return z_mu, z_sigma

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


class DIS(DiffusionProcess):
    def __init__(
        self,
        device: torch.device,
        num_steps: int,
        z_dim: int,
        h_dim: int,
        num_layers: int,
        non_linearity: str,
    ) -> None:
        super(DIS, self).__init__(device, num_steps)

        self.score_mlp = nn.Sequential(
            nn.Linear(z_dim + h_dim, h_dim),
            *[
                layer
                for layer in (getattr(nn, non_linearity)(), nn.Linear(h_dim, h_dim))
                for _ in range(num_layers - 1)
            ],
            nn.Linear(h_dim, z_dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(1, z_dim + h_dim), getattr(nn, non_linearity)()
        )

        self.betas = nn.ParameterList(
            [torch.tensor([t]) for t in torch.linspace(0.001, 10, num_steps).tolist()]
        )

        # self.betas = nn.ParameterList(
        #     [nn.Parameter(torch.tensor(1.0)) for _ in range(num_steps)]
        # )

        self.sigmas = nn.ParameterList([nn.Parameter(torch.ones(z_dim))])

    def forward_z_mu_and_z_sigma(
        self, z_prev: Tensor, t: int, h: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim), (1), (batch_size, h_dim)

        beta_t = self.betas[t]
        # (1)

        t_tensor = torch.tensor([t]).float().to(h.device)
        t_tensor = self.time_embedding(t_tensor).unsqueeze(0)
        t_tensor = t_tensor.repeat(z_prev.shape[0], 1)
        # (batch_size, z_dim)

        score = torch.cat([z_prev, h], dim=1)
        score = self.score_mlp(score + t_tensor)
        # (batch_size, z_dim)

        z_mu = z_prev + (beta_t * z_prev + score) * self.delta_t
        # (batch_size, z_dim)

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigmas[0]
        # (batch_size, z_dim)

        return z_mu, z_sigma

    def backward_z_mu_and_z_sigma(
        self, z_next: Tensor, t: int, h: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim), (1), (batch_size, h_dim)

        beta_t = self.betas[t]
        # (1)

        z_mu = (z_next - beta_t * z_next) * self.delta_t
        # (batch_size, z_dim)

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigmas[0]
        # (batch_size, z_dim)

        return z_mu, z_sigma
