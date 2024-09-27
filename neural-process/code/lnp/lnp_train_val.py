from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import wandb
from latent_neural_process import LatentNeuralProcess
from torch import Tensor
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.optim import Optimizer
from tqdm import tqdm
from utils import DataModule, split_context_target


def train_and_validate(
    model: LatentNeuralProcess,
    device: torch.device,
    data_module: DataModule,
    optimizer: Optimizer,
    num_epochs: int,
    preprocessing: Optional[
        Callable[[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
    ] = None,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
]:

    # torch.autograd.set_detect_anomaly(True)

    avg_train_neg_log_likes = []
    avg_train_kl_divs = []

    avg_val_neg_log_likes = []
    avg_val_kl_divs = []

    for epoch in range(num_epochs):

        model.train()
        with torch.inference_mode(False):

            train_neg_log_likes = []
            train_kl_divs = []

            dataloader = data_module.train_dataloader()
            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss, neg_log_like, kl_div = step(model, device, batch, preprocessing)

                optimizer.zero_grad()
                loss.backward()  # type: ignore
                optimizer.step()

                loop.set_postfix(
                    epoch=epoch,
                    loss=loss.item(),
                    recon_loss=neg_log_like.item(),
                    kl_div=kl_div.item(),
                )

                train_neg_log_likes.append(neg_log_like.item())
                train_kl_divs.append(kl_div.item())

                wandb.log(
                    {
                        "train_neg_log_like": neg_log_like.item(),
                        "train_kl_div": kl_div.item(),
                    }
                )

            avg_train_neg_log_like = float(np.mean(train_neg_log_likes))
            avg_train_kl_div = float(np.mean(train_kl_divs))

            avg_train_neg_log_likes.append(avg_train_neg_log_like)
            avg_train_kl_divs.append(avg_train_kl_div)

            wandb.log(
                {
                    "avg_train_neg_log_like": avg_train_neg_log_like,
                    "avg_train_kl_div": avg_train_kl_div,
                }
            )

        model.eval()
        with torch.no_grad():

            val_neg_log_likes = []
            val_kl_divs = []

            dataloader = data_module.val_dataloader()
            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss, neg_log_like, kl_div = step(model, device, batch, preprocessing)

                val_neg_log_likes.append(neg_log_like.item())
                val_kl_divs.append(kl_div.item())

                loop.set_postfix(
                    epoch=epoch,
                    loss=loss.item(),
                    recon_loss=neg_log_like.item(),
                    kl_div=kl_div.item(),
                )

                wandb.log(
                    {
                        "val_neg_log_like": neg_log_like.item(),
                        "val_kl_div": kl_div.item(),
                    }
                )

            avg_val_neg_log_like = float(np.mean(val_neg_log_likes))
            avg_val_kl_div = float(np.mean(val_kl_divs))

            avg_val_neg_log_likes.append(avg_val_neg_log_like)
            avg_val_kl_divs.append(avg_val_kl_div)

            wandb.log(
                {
                    "avg_val_neg_log_like": avg_val_neg_log_like,
                    "avg_val_kl_div": avg_val_kl_div,
                }
            )

    return (
        avg_train_neg_log_likes,
        avg_train_kl_divs,
        avg_val_neg_log_likes,
        avg_val_kl_divs,
    )


def step(
    model: LatentNeuralProcess,
    device: torch.device,
    batch: Tuple[Tensor, Tensor],
    preprocessing: Optional[
        Callable[[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
    ] = None,
) -> Tuple[Tensor, Tensor, Tensor]:

    x_data, y_data = batch if preprocessing is None else preprocessing(batch)
    x_data, y_data = x_data.to(device), y_data.to(device)

    factor = max(min(0.9, np.random.random()), 0.1)
    x_context, y_context, x_target, y_target = split_context_target(
        x_data, y_data, factor
    )

    z, mu_D, logvar_D = model.encode(x_data, y_data)
    _, mu_C, logvar_C = model.encode(x_context, y_context)
    mu, logvar = model.decode(z, x_data)

    D_distro = Normal(mu_D, torch.exp(0.5 * logvar_D))  # type: ignore
    C_distro = Normal(mu_C, torch.exp(0.5 * logvar_C))  # type: ignore
    pred_distro = Normal(mu, torch.exp(0.5 * logvar))  # type: ignore

    neg_log_like = -pred_distro.log_prob(y_data).mean(dim=0).sum()  # type: ignore
    kl_div = kl_divergence(D_distro, C_distro).mean(dim=0).sum()
    loss = neg_log_like + kl_div

    return loss, neg_log_like, kl_div
