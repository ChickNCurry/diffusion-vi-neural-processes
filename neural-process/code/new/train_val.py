from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from components import NeuralProcess
from torch import Tensor, nn
from torch.optim import Optimizer
from tqdm import tqdm
from utils import DataModule, split_context_target


def train_and_validate(
    model: NeuralProcess,
    device: torch.device,
    data_module: DataModule,
    optimizer: Optimizer,
    recon_criterion: nn.Module,
    kl_reduction: str,
    num_epochs: int,
    preprocessing: Optional[Callable[[Tensor], Tuple[Tensor, Tensor]]] = None,
) -> Tuple[
    List[np.floating[Any]],
    List[np.floating[Any]],
    List[np.floating[Any]],
    List[np.floating[Any]],
]:

    # torch.autograd.set_detect_anomaly(True)

    avg_train_recon_losses = []
    avg_train_kl_divs = []

    avg_val_recon_losses = []
    avg_val_kl_divs = []

    for epoch in range(num_epochs):

        model.train()
        with torch.inference_mode(False):

            train_recon_losses = []
            train_kl_divs = []

            dataloader = data_module.train_dataloader()
            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss, recon_loss, kl_div = step(
                    model, device, batch, recon_criterion, kl_reduction, preprocessing
                )

                optimizer.zero_grad()
                loss.backward()  # type: ignore
                optimizer.step()

                loop.set_postfix(
                    epoch=epoch,
                    recon_loss=recon_loss.item(),
                    kl_div=kl_div.item(),
                )

                train_recon_losses.append(recon_loss.item())
                train_kl_divs.append(kl_div.item())

            avg_train_recon_losses.append(np.mean(train_recon_losses))
            avg_train_kl_divs.append(np.mean(train_kl_divs))

        model.eval()
        with torch.no_grad():

            val_recon_losses = []
            val_kl_divs = []

            dataloader = data_module.val_dataloader()
            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss, recon_loss, kl_div = step(
                    model, device, batch, recon_criterion, kl_reduction, preprocessing
                )

                loop.set_postfix(
                    epoch=epoch,
                    recon_loss=recon_loss.item(),
                    kl_div=kl_div.item(),
                )

                val_recon_losses.append(recon_loss.item())
                val_kl_divs.append(kl_div.item())

            avg_val_recon_losses.append(np.mean(val_recon_losses))
            avg_val_kl_divs.append(np.mean(val_kl_divs))

    return (
        avg_train_recon_losses,
        avg_train_kl_divs,
        avg_val_recon_losses,
        avg_val_kl_divs,
    )


def step(
    model: NeuralProcess,
    device: torch.device,
    batch: Tuple[Tensor, Tensor],
    recon_criterion: nn.Module,
    kl_reduction: str,
    preprocessing: Optional[Callable[[Tensor], Tuple[Tensor, Tensor]]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:

    x_data: Tensor
    y_data: Tensor

    if preprocessing is not None:
        x_data, y_data = preprocessing(batch[0])
    else:
        x_data, y_data = batch

    x_data = x_data.to(device)
    y_data = y_data.to(device)

    factor = max(min(0.9, np.random.random()), 0.1)
    x_context, y_context, x_target, y_target = split_context_target(
        x_data, y_data, factor
    )
    # -> (batch_size, context_size, x_dim + y_dim), (batch_size, target_size, x_dim + y_dim)

    z, mu_D, logvar_D = model.encode(x_data, y_data)
    _, mu_C, logvar_C = model.encode(x_context, y_context)

    mu, logvar = model.decode(z, x_target)

    recon_loss: Tensor = recon_criterion(mu, y_target)
    kl_div = kl_divergence(mu_D, logvar_D, mu_C, logvar_C, reduction=kl_reduction)
    loss = recon_loss + kl_div

    return loss, recon_loss, kl_div


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
