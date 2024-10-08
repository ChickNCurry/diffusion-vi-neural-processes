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

    def reparameterize(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
        # (batch_size, z_dim),
        # (batch_size, z_dim)

        eps = torch.randn_like(z_sigma)
        # (batch_size, z_dim)

        z = z_mu + z_sigma * eps
        # (batch_size, z_dim)

        return z

    @abstractmethod
    def get_z_mu_and_z_sigma(
        self, z: Tensor, t: int, h: Tensor
    ) -> Tuple[Tensor, Tensor]:
        pass

    def forward_transition(
        self, z_prev: Tensor, t: int, z_r: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, z_dim), (1), (batch_size, context_size, x_dim)

        z_mu, z_sigma = self.get_z_mu_and_z_sigma(z_prev, t, z_r)
        # (batch_size, z_dim), (batch_size, z_dim)

        z = self.reparameterize(z_mu, z_sigma)
        # (batch_size, z_dim)

        return z, z_mu, z_sigma

    # @abstractmethod
    # def backward_mean_var(self, z: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    #     pass

    # def backward_transition(
    #     self, z: Tensor, t: Tensor
    # ) -> Tuple[Tensor, Tensor, Tensor]:
    #     # (batch_size, z_dim), (1)

    #     z_mean = self.backward_mean(z, t)
    #     # (batch_size, z_dim)

    #     z_var = self.backward_var(t)
    #     # (1)

    #     z_prev = self.reparameterize(z_mean, z_var)
    #     # (batch_size, z_dim)

    #     return z_prev, z_mean, torch.sqrt(z_var)


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

    def get_z_mu_and_z_sigma(
        self, z: Tensor, t: int, h: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # (batch_size, z_dim), (1), (batch_size, h_dim)

        beta_t = self.betas[t]
        # (1)

        t_tensor = torch.tensor([t]).float().to(h.device)
        t_tensor = self.time_embedding(t_tensor).unsqueeze(0)
        # (1, z_dim)

        t_tensor = t_tensor.repeat(z.shape[0], 1)
        # (batch_size, z_dim)

        score = torch.cat([z, h], dim=1)
        score = self.score_mlp(score + t_tensor)
        # (batch_size, z_dim)

        z_mu = z + (beta_t * z + score) * self.delta_t
        # (batch_size, z_dim)

        z_sigma = torch.sqrt(2 * beta_t * self.delta_t) * self.sigmas[0]
        # (batch_size, z_dim)

        return z_mu, z_sigma


# class DDS(DiffusionProcess):
#     def __init__(
#         self,
#         device: torch.device,
#         num_steps: int,
#         z_dim: int,
#         h_dim: int,
#         beta_start: float,
#         beta_end: float,
#         std_0: Tensor,
#     ) -> None:
#         super(DDS, self).__init__(device, num_steps)

#         self.std_0 = std_0.to(device)

#         self.sore_function_mlp = nn.Sequential(
#             nn.Linear(z_dim, h_dim),
#             nn.SiLU(),
#             nn.Linear(h_dim, h_dim),
#             nn.SiLU(),
#             nn.Linear(h_dim, z_dim),
#         )

#         self.time_embedding = nn.Sequential(
#             nn.Linear(1, z_dim),
#             nn.Tanh(),
#         )

#         self.betas = torch.linspace(
#             beta_start, beta_end, num_steps, dtype=torch.float32
#         ).to(device)

#     def forward_mean_var(
#         self, z: Tensor, t: Tensor, h: Tensor
#     ) -> Tuple[Tensor, Tensor]:
#         # (batch_size, z_dim), (1), (1)

#         t = self.time_embedding(t.float()).unsqueeze(0)
#         # (1, z_dim)

#         t = t.repeat(z.shape[0], 1)
#         # (batch_size, z_dim)

#         s: Tensor = self.sore_function_mlp(z + t)
#         # (batch_size, z_dim)

#         beta_t = self.betas[t.int()]
#         # (1)

#         z_mean = (torch.sqrt(1 - beta_t) * z + s) * self.delta_t
#         # (batch_size, z_dim)

#         z_var = beta_t * torch.pow(self.std_0, 2) * self.delta_t
#         # (1)

#         return z_mean, z_var

# def backward_mean_var(self, z: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
#     # (batch_size, z_dim), (1), (1)

#     beta_t = self.betas[t.int()]
#     # (1)

#     z_mean = torch.sqrt(1 - beta_t) * z * self.delta_t
#     # (batch_size, z_dim)

#     z_var = beta_t * torch.pow(self.std_0, 2) * self.delta_t
#     # (1)

#     return z_mean, z_var
