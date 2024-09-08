import torch
from torch import Tensor, nn


class DDPM(nn.Module):
    def __init__(
        self,
        device: torch.device,
        network: nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super(DDPM, self).__init__()

        self.device = device
        self.network = network
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, dtype=torch.float32
        ).to(device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def add_noise(self, x_start: Tensor, x_noise: Tensor, timesteps: Tensor) -> Tensor:
        # The forward process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps]  # bs
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]  # bs
        s1 = s1.reshape(-1, 1, 1, 1)  # (bs, 1, 1, 1) for broadcasting
        s2 = s2.reshape(-1, 1, 1, 1)  # (bs, 1, 1, 1)
        return s1 * x_start + s2 * x_noise

    def reverse(self, x: Tensor, t: Tensor) -> Tensor:
        # The network return the estimation of the noise we added
        out: Tensor = self.network(x, t)
        return out

    def step(self, model_output: Tensor, timestep: Tensor, sample: Tensor) -> Tensor:
        # one step of sampling
        # timestep (1)
        t = timestep

        coef_epsilon = (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod
        coef_eps_t = coef_epsilon[t].reshape(-1, 1, 1, 1)

        coef_first = torch.sqrt(1 / self.alphas)
        coef_first_t = coef_first[t].reshape(-1, 1, 1, 1)

        pred_prev_sample: Tensor = coef_first_t * (sample - coef_eps_t * model_output)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(self.device)
            variance = (self.betas[t] ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
