from typing import Callable, Optional, Tuple

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
) -> None:

    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_epochs):

        neural_process.train()
        with torch.inference_mode(False):

            dataloader = data_module.train_dataloader()
            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss, log_like, diffu_loss = step(
                    neural_process, device, batch, preprocessing
                )

                optimizer.zero_grad()
                loss.backward()  # type: ignore
                optimizer.step()

                loop.set_postfix(
                    epoch=epoch,
                    loss=loss.item(),
                    log_like=log_like.item(),
                    diffu_loss=diffu_loss.item(),
                )

                if wandb_logging:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/log_like": log_like.item(),
                            "train/diffu_loss": diffu_loss.item(),
                        }
                    )

        if not validate:
            continue

        neural_process.eval()
        with torch.no_grad():

            dataloader = data_module.val_dataloader()
            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss, log_like, diffu_loss = step(
                    neural_process, device, batch, preprocessing
                )

                loop.set_postfix(
                    epoch=epoch,
                    loss=loss.item(),
                    log_like=log_like.item(),
                    diffu_loss=diffu_loss.item(),
                )

                if wandb_logging:
                    wandb.log(
                        {
                            "val/loss": loss.item(),
                            "val/log_like": log_like.item(),
                            "val/diffu_loss": diffu_loss.item(),
                        }
                    )


def step(
    neural_process: NeuralProcess,
    device: torch.device,
    batch: Tuple[Tensor, Tensor],
    preprocessing: Optional[
        Callable[[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
    ] = None,
) -> Tuple[Tensor, Tensor, Tensor]:

    x_data, y_data = batch if preprocessing is None else preprocessing(batch)
    x_data, y_data = x_data.to(device), y_data.to(device)

    context_len = int(max(1, np.random.random() * x_data.shape[1]))
    x_context, y_context, _, _ = split_context_target(x_data, y_data, context_len)

    output, z_tuples_D = neural_process.encode(x_data, y_data, x_data)
    _, z_tuples_C = neural_process.encode(x_context, y_context, x_data)
    _, y_mu, y_std = neural_process.decode(x_data, output)

    log_like = Normal(y_mu, y_std).log_prob(y_data).mean(dim=0).sum()  # type: ignore

    # priors_kl = (
    #     kl_divergence(
    #         Normal(z_tuples_D[0].z_mu, z_tuples_D[0].z_sigma),  # type: ignore
    #         Normal(z_tuples_C[0].z_mu, z_tuples_C[0].z_sigma),  # type: ignore
    #     )
    #     .mean(dim=0)
    #     .sum()
    # )

    # diffu_kl = (
    #     torch.stack(
    #         [
    #             kl_divergence(
    #                 Normal(z_tuples_D[i].z_mu, z_tuples_D[i].z_sigma),  # type: ignore
    #                 Normal(z_tuples_C[i].z_mu, z_tuples_C[i].z_sigma),  # type: ignore
    #             )
    #             .mean(dim=0)
    #             .sum()
    #             for i in range(1, len(z_tuples_D))
    #         ]
    #     ).sum()
    #     if len(z_tuples_D) > 1
    #     else torch.tensor(0.0)
    # )

    # loss = -log_like + priors_kl + diffu_kl

    # return loss, -log_like, priors_kl + diffu_kl

    forward_log_like_C = torch.stack(
        [
            Normal(z_tuples_C[i].z_mu, z_tuples_C[i].z_sigma).log_prob(z_tuples_D[i].z).mean(dim=0).sum()  # type: ignore
            for i in range(len(z_tuples_C))
        ]
    ).sum()

    forward_log_like_D = torch.stack(
        [
            Normal(z_tuples_D[i].z_mu, z_tuples_D[i].z_sigma).log_prob(z_tuples_D[i].z).mean(dim=0).sum()  # type: ignore
            for i in range(len(z_tuples_C))
        ]
    ).sum()

    loss = -log_like - forward_log_like_C + forward_log_like_D

    return loss, -log_like, -forward_log_like_C + forward_log_like_D


def step_new(
    neural_process: NeuralProcess,
    device: torch.device,
    batch: Tuple[Tensor, Tensor],
    preprocessing: Optional[
        Callable[[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
    ] = None,
) -> Tuple[Tensor, Tensor, Tensor]:

    x_data, y_data = batch if preprocessing is None else preprocessing(batch)
    x_data, y_data = x_data.to(device), y_data.to(device)

    context_len = int(max(1, np.random.random() * x_data.shape[1]))
    x_context, y_context, _, _ = split_context_target(x_data, y_data, context_len)

    output_D, z_tuples_D = neural_process.encode(x_data, y_data, x_data)
    # output_C, z_tuples_C = neural_process.encode(x_context, y_context, x_context)

    r_c = neural_process.encoder.latent_encoder.encoder(x_context, y_context, x_data)  # type: ignore
    z_tuples_C = neural_process.encoder.latent_encoder.backward_process(r_c, z_tuples_D)  # type: ignore

    _, y_mu, y_std = neural_process.decode(x_data, output_D)
    # _, y_mu_context, y_std_context = neural_process.decode(x_context, output_C)

    log_like = Normal(y_mu, y_std).log_prob(y_data).mean(dim=0).sum()  # type: ignore
    # log_like += (
    #     Normal(y_mu_context, y_std_context).log_prob(y_context).mean(dim=0).sum()  # type: ignore
    # )

    forward_log_like = torch.stack(
        [
            Normal(z_tuples_D[i].z_mu, z_tuples_D[i].z_sigma).log_prob(z_tuples_D[i].z).mean(dim=0).sum()  # type: ignore
            for i in range(len(z_tuples_D))
        ]
    ).sum()

    backward_log_like = torch.stack(
        [
            Normal(z_tuples_C[i].z_mu, z_tuples_C[i].z_sigma).log_prob(z_tuples_C[i].z).mean(dim=0).sum()  # type: ignore
            for i in range(len(z_tuples_C) - 1)
        ]
    ).sum()

    loss = -log_like - backward_log_like + forward_log_like

    return loss, -log_like, -backward_log_like + forward_log_like
