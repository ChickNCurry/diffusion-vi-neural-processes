from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import wandb
from neural_process import NeuralProcess
from torch import Tensor
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.optim import Optimizer
from tqdm import tqdm
from utils import DataModule, split_context_target


def train_and_validate(
    model: NeuralProcess,
    device: torch.device,
    data_module: DataModule,
    optimizer: Optimizer,
    num_epochs: int,
    preprocessing: Optional[
        Callable[[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
    ] = None,
    wandb_logging: bool = True,
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

                if wandb_logging:
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

            if wandb_logging:
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

                if wandb_logging:
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

            if wandb_logging:
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
    model: NeuralProcess,
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

    r, z, z_mu_D, z_std_D = model.encode(x_data, y_data)
    _, _, z_mu_C, z_std_C = model.encode(x_context, y_context)
    y, y_mu, y_std = model.decode(x_data, r, z)

    z_distro_D = Normal(z_mu_D, z_std_D)  # type: ignore
    z_distro_C = Normal(z_mu_C, z_std_C)  # type: ignore
    y_distro = Normal(y_mu, y_std)  # type: ignore

    neg_log_like = -y_distro.log_prob(y_data).mean(dim=0).sum()  # type: ignore
    kl_div = kl_divergence(z_distro_D, z_distro_C).mean(dim=0).sum()
    loss = neg_log_like + kl_div

    return loss, neg_log_like, kl_div
