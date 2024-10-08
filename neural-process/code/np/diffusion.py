from abc import ABC, abstractmethod
from typing import Tuple

import torch
from neural_process import NeuralProcess
from torch import Tensor, nn


class DiffusionVariant(nn.Module, ABC):
    def __init__(self, device: torch.device, num_steps: int) -> None:
        super(DiffusionVariant, self).__init__()

        self.num_steps = num_steps
        self.delta_t = (
            torch.tensor(1.0 / num_steps).to(device) if num_steps > 0 else None
        )

    @abstractmethod
    def forward_mean(self, z: Tensor, t: Tensor) -> Tensor:
        pass

    @abstractmethod
    def forward_var(self, t: Tensor) -> Tensor:
        pass

    @abstractmethod
    def backward_mean(self, z: Tensor, t: Tensor) -> Tensor:
        pass

    @abstractmethod
    def backward_var(self, t: Tensor) -> Tensor:
        pass

    def reparameterize(self, mean: Tensor, var: Tensor) -> Tensor:
        # (batch_size, z_dim), (1)

        eps = torch.randn_like(mean)
        # (batch_size, z_dim)

        z = mean + eps * torch.sqrt(var)
        # (batch_size, z_dim)

        return z

    def forward_transition(self, z: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, z_dim), (1)

        z_mean = self.forward_mean(z, t)
        # (batch_size, z_dim)

        z_var = self.forward_var(t)
        # (1)

        z_next = self.reparameterize(z_mean, z_var)
        # (batch_size, z_dim)

        return z_next, z_mean, torch.sqrt(z_var)

    def backward_transition(
        self, z: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, z_dim), (1)

        z_mean = self.backward_mean(z, t)
        # (batch_size, z_dim)

        z_var = self.backward_var(t)
        # (1)

        z_prev = self.reparameterize(z_mean, z_var)
        # (batch_size, z_dim)

        return z_prev, z_mean, torch.sqrt(z_var)


class DiffusionNeuralProcess(nn.Module):
    def __init__(self, np_model: NeuralProcess, diffu_model: DiffusionVariant) -> None:
        super(DiffusionNeuralProcess, self).__init__()

        self.np_model = np_model
        self.diffu_model = diffu_model

    def forward(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:

        r, z, _, _ = self.np_model.encode(x_context, y_context, x_target)

        for t in range(self.diffu_model.num_steps):
            z, _, _ = self.diffu_model.forward_transition(
                z, torch.tensor([t]).to(x_context.device)
            )

        y, y_mu, y_std = self.np_model.decode(x_target, r, z)

        return y, y_mu, y_std

    def sample(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor, num_samples: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        x_context = x_context.repeat(num_samples, 1, 1)
        y_context = y_context.repeat(num_samples, 1, 1)
        x_target = x_target.repeat(num_samples, 1, 1)

        r, z, z_mu, z_std = self.np_model.encode(x_context, y_context, x_target)

        for t in range(self.diffu_model.num_steps):
            z, z_mu, z_std = self.diffu_model.forward_transition(
                z, torch.tensor([t]).to(x_context.device)
            )

        y, y_mu, y_std = self.np_model.decode(x_target, r, z)

        return y, y_mu, y_std, z, z_mu, z_std


class DDS(DiffusionVariant):
    def __init__(
        self,
        device: torch.device,
        num_steps: int,
        z_dim: int,
        h_dim: int,
        beta_start: float,
        beta_end: float,
        std_0: Tensor,
    ) -> None:
        super(DDS, self).__init__(device, num_steps)

        self.std_0 = std_0.to(device)

        self.sore_function_mlp = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, z_dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(1, z_dim),
            nn.Tanh(),
        )

        self.betas = torch.linspace(
            beta_start, beta_end, num_steps, dtype=torch.float32
        ).to(device)

    def forward_mean(self, z: Tensor, t: Tensor) -> Tensor:
        # (batch_size, z_dim), (1), (1)

        t = self.time_embedding(t.float()).unsqueeze(0)
        # (1, z_dim)

        t = t.repeat(z.shape[0], 1)
        # (batch_size, z_dim)

        s: Tensor = self.sore_function_mlp(z + t)
        # (batch_size, z_dim)

        beta_t = self.betas[t.int()]
        # (1)

        z_mean = (torch.sqrt(1 - beta_t) * z + s) * self.delta_t
        # (batch_size, z_dim)

        return z_mean

    def forward_var(self, t: Tensor) -> Tensor:
        # (1), (1)

        beta_t = self.betas[t.int()]
        # (1)

        z_var = beta_t * torch.pow(self.std_0, 2) * self.delta_t
        # (1)

        return z_var

    def backward_mean(self, z: Tensor, t: Tensor) -> Tensor:
        # (batch_size, z_dim), (1), (1)

        beta_t = self.betas[t.int()]
        # (1)

        z_mean = torch.sqrt(1 - beta_t) * z * self.delta_t
        # (batch_size, z_dim)

        return z_mean

    def backward_var(self, t: Tensor) -> Tensor:
        # (1), (1)

        beta_t = self.betas[t.int()]
        # (1)

        z_var = beta_t * torch.pow(self.std_0, 2) * self.delta_t
        # (1)

        return z_var
