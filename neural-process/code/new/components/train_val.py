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
    neural_process: NeuralProcess,
    device: torch.device,
    data_module: DataModule,
    optimizer: Optimizer,
    num_epochs: int,
    wandb_logging: bool = True,
    preprocessing: Optional[
        Callable[[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
    ] = None,
    validate: bool = False,
) -> Tuple[
    List[float], List[float], List[float], List[float], List[float], List[float]
]:

    # torch.autograd.set_detect_anomaly(True)

    avg_train_log_likes = []
    avg_train_priors_kls = []
    avg_train_diffu_kls = []

    avg_val_log_likes = []
    avg_val_priors_kls = []
    avg_val_diffu_kls = []

    for epoch in range(num_epochs):

        neural_process.train()
        with torch.inference_mode(False):

            train_log_likes = []
            train_priors_kls = []
            train_diffu_kls = []

            dataloader = data_module.train_dataloader()
            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss, log_like, priors_kl, diffu_kl = step(
                    neural_process, device, batch, preprocessing
                )

                optimizer.zero_grad()
                loss.backward()  # type: ignore
                optimizer.step()

                loop.set_postfix(
                    epoch=epoch,
                    loss=loss.item(),
                    log_like=log_like.item(),
                    priors_kl=priors_kl.item(),
                    diffu_kl=diffu_kl.item(),
                )

                train_log_likes.append(log_like.item())
                train_priors_kls.append(priors_kl.item())
                train_diffu_kls.append(diffu_kl.item())

                if wandb_logging:
                    wandb.log(
                        {
                            "train/log_like": log_like.item(),
                            "train/priors_kl": priors_kl.item(),
                            "train/diffu_kl": diffu_kl.item(),
                            "train/loss": log_like.item()
                            + priors_kl.item()
                            + diffu_kl.item(),
                        }
                    )

            avg_train_log_like = float(np.mean(train_log_likes))
            avg_train_priors_kl = float(np.mean(train_priors_kls))
            avg_train_diffu_kl = float(np.mean(train_diffu_kls))

            avg_train_log_likes.append(avg_train_log_like)
            avg_train_priors_kls.append(avg_train_priors_kl)
            avg_train_diffu_kls.append(avg_train_diffu_kl)

        if not validate:
            continue

        neural_process.eval()
        with torch.no_grad():

            val_log_likes = []
            val_priors_kls = []
            val_diffu_kls = []

            dataloader = data_module.val_dataloader()
            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss, log_like, priors_kl, diffu_kl = step(
                    neural_process, device, batch, preprocessing
                )

                val_log_likes.append(log_like.item())
                val_priors_kls.append(priors_kl.item())
                val_diffu_kls.append(diffu_kl.item())

                loop.set_postfix(
                    epoch=epoch,
                    loss=loss.item(),
                    log_like=log_like.item(),
                    priors_kl=priors_kl.item(),
                    diffu_kl=diffu_kl.item(),
                )

                if wandb_logging:
                    wandb.log(
                        {
                            "val/log_like": log_like.item(),
                            "val/priors_kl": priors_kl.item(),
                            "val/diffu_kl": diffu_kl.item(),
                            "val/loss": log_like.item()
                            + priors_kl.item()
                            + diffu_kl.item(),
                        }
                    )

            avg_val_log_like = float(np.mean(val_log_likes))
            avg_val_priors_kl = float(np.mean(val_priors_kls))
            avg_val_diffu_kl = float(np.mean(val_diffu_kls))

            avg_val_log_likes.append(avg_val_log_like)
            avg_val_priors_kls.append(avg_val_priors_kl)
            avg_val_diffu_kls.append(avg_val_diffu_kl)

    return (
        avg_train_log_likes,
        avg_train_priors_kls,
        avg_train_diffu_kls,
        avg_val_log_likes,
        avg_val_priors_kls,
        avg_val_diffu_kls,
    )


def step(
    neural_process: NeuralProcess,
    device: torch.device,
    batch: Tuple[Tensor, Tensor],
    preprocessing: Optional[
        Callable[[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
    ] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    x_data, y_data = batch if preprocessing is None else preprocessing(batch)
    x_data, y_data = x_data.to(device), y_data.to(device)

    context_len = int(max(1, np.random.random() * x_data.shape[1]))
    x_context, y_context, _, _ = split_context_target(x_data, y_data, context_len)

    output, z_tuples_D = neural_process.encode(x_data, y_data, x_data)
    _, z_tuples_C = neural_process.encode(x_context, y_context, x_data)
    _, y_mu, y_std = neural_process.decode(x_data, output)

    log_like = Normal(y_mu, y_std).log_prob(y_data).mean(dim=0).sum()  # type: ignore

    priors_kl = (
        kl_divergence(
            Normal(z_tuples_D[0].z_mu, z_tuples_D[0].z_sigma),  # type: ignore
            Normal(z_tuples_C[0].z_mu, z_tuples_C[0].z_sigma),  # type: ignore
        )
        .mean(dim=0)
        .sum()
    )

    diffu_kl = (
        torch.stack(
            [
                kl_divergence(
                    Normal(z_tuples_D[i].z_mu, z_tuples_D[i].z_sigma),  # type: ignore
                    Normal(z_tuples_C[i].z_mu, z_tuples_C[i].z_sigma),  # type: ignore
                )
                .mean(dim=0)
                .sum()
                for i in range(1, len(z_tuples_D))
            ]
        ).sum()
        if len(z_tuples_D) > 1
        else torch.tensor(0.0)
    )

    loss = -log_like + priors_kl + diffu_kl

    return loss, log_like, priors_kl, diffu_kl
