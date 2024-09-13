from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm


def visualize_batch(
    batch: Tensor,
) -> None:
    plt.figure(figsize=(9, 3))

    for i in range(batch[0].shape[0]):
        x = batch[0][i, :, 0]
        y = batch[1][i, :, 0]
        plt.scatter(x, y)
    plt.show()


def visualize_decoder(model: torch.nn.Module, device: torch.device, z_dim: int) -> None:
    plt.figure(figsize=(9, 3))

    x_target = torch.Tensor(np.linspace(-5, 5, 100)).to(device)
    x_target = x_target.unsqueeze(1).repeat(10, 1, 1)
    z_sample = torch.randn(10, z_dim).to(device)

    mu, logvar = model.decode(z_sample, x_target)

    for i in range(x_target.shape[0]):
        x = x_target[i, :, 0].detach().cpu().numpy()
        y = mu[i, :, 0].detach().cpu().numpy()
        plt.scatter(x, y)
    plt.show()


def sample_latent_space(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,  # type: ignore
    num_samples: int = 1000,
) -> Tuple[List[Tensor], List[Tensor]]:
    mus = []
    logvars = []

    for i, (batch) in tqdm(enumerate(dataloader), total=num_samples):
        sequence = torch.cat(batch, dim=-1).to(device)
        _, mu, logvar = model.encode(sequence)

        mus.append(mu.cpu().detach().numpy())
        logvars.append(logvar.cpu().detach().numpy())

        if i > num_samples:
            break

    return mus, logvars


def visualize_latent_space(
    mus: List[Tensor],
    logvars: List[Tensor],
    dim_1: int,
    dim_2: int,
) -> None:
    plt.figure(figsize=(9, 3))
    plt.scatter([m[0][dim_1] for m in mus], [m[0][dim_2] for m in mus])
    plt.show()


def visualize_losses(recon_losses: List[float], kl_divs: List[float]) -> None:
    _, ax = plt.subplots(1, 2, figsize=(15, 3))

    ax[0].plot(recon_losses)
    ax[0].set_title("Reconstruction Loss")
    ax[0].grid()

    ax[1].plot(kl_divs)
    ax[1].set_title("KL Divergence")
    ax[1].grid()

    plt.tight_layout()
    plt.show()
