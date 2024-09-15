from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import split_context_target, img_batch_to_sequence


def train(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,  # type: ignore
    optimizer: Optimizer,
    recon_criterion: nn.Module,
    num_epochs: int = 80,
) -> Tuple[List[float], List[float]]:

    # torch.autograd.set_detect_anomaly(True)

    recon_losses = []
    kl_divs = []

    model.train()
    with torch.inference_mode(False):

        for epoch in range(num_epochs):

            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                # forward
                factor = max(min(0.9, np.random.random()), 0.1)
                sequence = torch.cat(batch, dim=-1).to(device)
                context, target = split_context_target(sequence, factor)

                x_target = target[:, :, 0:1]
                y_target = target[:, :, 1:2]

                z, mu_D, logvar_D = model.encode(sequence)
                _, mu_C, logvar_C = model.encode(context)

                mu, logvar = model.decode(z, x_target)

                # loss
                recon_loss = recon_criterion(mu, y_target)
                kl_div = kl_divergence(mu_D, logvar_D, mu_C, logvar_C, reduction="mean")
                loss = recon_loss + kl_div

                # stats
                recon_losses.append(recon_loss.item())
                kl_divs.append(kl_div.item())
                loop.set_postfix(
                    epoch=epoch,
                    factor=factor,
                    loss=loss.item(),
                    recon_loss=recon_loss.item(),
                    kl_div=kl_div.item(),
                )

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return recon_losses, kl_divs


def train_2d(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,  # type: ignore
    optimizer: Optimizer,
    recon_criterion: nn.Module,
    num_epochs: int = 80,
) -> Tuple[List[float], List[float]]:

    # torch.autograd.set_detect_anomaly(True)

    recon_losses = []
    kl_divs = []

    model.train()
    with torch.inference_mode(False):

        for epoch in range(num_epochs):

            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                # forward
                factor = max(min(0.9, np.random.random()), 0.1)
                sequence = img_batch_to_sequence(batch[0]).to(device)
                context, target = split_context_target(sequence, factor)

                x_target = target[:, :, 0:1]
                y_target = target[:, :, 1:2]

                z, mu_D, logvar_D = model.encode(sequence)
                _, mu_C, logvar_C = model.encode(context)

                mu, logvar = model.decode(z, x_target)

                # loss
                recon_loss = recon_criterion(mu, y_target)
                kl_div = kl_divergence(mu_D, logvar_D, mu_C, logvar_C, reduction="mean")
                loss = recon_loss + kl_div

                # stats
                recon_losses.append(recon_loss.item())
                kl_divs.append(kl_div.item())
                loop.set_postfix(
                    epoch=epoch,
                    factor=factor,
                    loss=loss.item(),
                    recon_loss=recon_loss.item(),
                    kl_div=kl_div.item(),
                )

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return recon_losses, kl_divs


def kl_divergence(
    mu_1: Tensor,
    logvar_1: Tensor,
    mu_2: Tensor,
    logvar_2: Tensor,
    reduction: str = "mean",
) -> Tensor:
    # (batch_size, z_dim)

    k = mu_1.size(1)

    term1 = (logvar_1.exp() / logvar_2.exp()).sum(1)
    term2 = (logvar_2 - logvar_1).sum(1)
    term3 = ((mu_1 - mu_2).pow(2) / logvar_2.exp()).sum(1)

    kl = torch.tensor(0.0)

    match reduction:
        case "mean":
            kl = 0.5 * torch.mean(term1 + term2 + term3 - k)
        case "sum":
            kl = 0.5 * torch.sum(term1 + term2 + term3 - k)

    return kl


# def kl_criterion(mu_D, logvar_D, mu_C, logvar_C, eps=1e-6):
#     # (batch_size, z_dim)

#     # Clipping log variances to prevent numerical instabilities
#     logvar_D_clipped = torch.clamp(logvar_D, min=-10.0, max=10.0)
#     logvar_C_clipped = torch.clamp(logvar_C, min=-10.0, max=10.0)

#     k = mu_D.size(1)

#     # Compute terms with added stability
#     term1 = (logvar_D_clipped.exp() / (logvar_C_clipped.exp() + eps)).sum(1)
#     term2 = (logvar_C_clipped - logvar_D_clipped).sum(1)
#     term3 = ((mu_D - mu_C).pow(2) / (logvar_C_clipped.exp() + eps)).sum(1)

#     kl = 0.5 * (term1 + term2 + term3 - k) # Per-sample KL divergence

#     return torch.sum(kl)
