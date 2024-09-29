from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import wandb
from diffusion import DiffusionNeuralProcess
from torch import Tensor
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.optim import Optimizer
from tqdm import tqdm
from utils import DataModule, split_context_target


def train_and_validate(
    diffu_vi_np_model: DiffusionNeuralProcess,
    device: torch.device,
    data_module: DataModule,
    optimizer: Optimizer,
    num_epochs: int,
    wandb_logging: bool = True,
    preprocessing: Optional[
        Callable[[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
    ] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:

    # torch.autograd.set_detect_anomaly(True)

    avg_train_log_likes = []
    avg_train_diffu_losses = []

    avg_val_log_likes = []
    avg_val_diffu_losses = []

    for epoch in range(num_epochs):

        diffu_vi_np_model.train()
        with torch.inference_mode(False):

            train_log_likes = []
            train_diffu_losses = []

            dataloader = data_module.train_dataloader()
            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss, log_like, diffu_loss = step(
                    diffu_vi_np_model, device, batch, preprocessing
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

                train_log_likes.append(log_like.item())
                train_diffu_losses.append(diffu_loss.item())

                if wandb_logging:
                    wandb.log(
                        {
                            "train/log_like": log_like.item(),
                            "train/diffu_loss": diffu_loss.item(),
                        }
                    )

            avg_train_log_like = float(np.mean(train_log_likes))
            avg_train_diffu_loss = float(np.mean(train_diffu_losses))

            avg_train_log_likes.append(avg_train_log_like)
            avg_train_diffu_losses.append(avg_train_diffu_loss)

        diffu_vi_np_model.eval()
        with torch.no_grad():

            val_log_likes = []
            val_diffu_losses = []

            dataloader = data_module.val_dataloader()
            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss, log_like, diffu_loss = step(
                    diffu_vi_np_model, device, batch, preprocessing
                )

                val_log_likes.append(log_like.item())
                val_diffu_losses.append(diffu_loss.item())

                loop.set_postfix(
                    epoch=epoch,
                    loss=loss.item(),
                    log_like=log_like.item(),
                    diffu_loss=diffu_loss.item(),
                )

                if wandb_logging:
                    wandb.log(
                        {
                            "val/log_like": log_like.item(),
                            "val/diffu_loss": diffu_loss.item(),
                        }
                    )

            avg_val_log_like = float(np.mean(val_log_likes))
            avg_val_diffu_loss = float(np.mean(val_diffu_losses))

            avg_val_log_likes.append(avg_val_log_like)
            avg_val_diffu_losses.append(avg_val_diffu_loss)

    return (
        avg_train_log_likes,
        avg_train_diffu_losses,
        avg_val_log_likes,
        avg_val_diffu_losses,
    )


def step(
    diffu_vi_np_model: DiffusionNeuralProcess,
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

    r, z_D_0, z_mu_D_0, z_std_D_0 = diffu_vi_np_model.np_model.encode(
        x_data, y_data, x_data
    )
    _, z_C_0, z_mu_C_0, z_std_C_0 = diffu_vi_np_model.np_model.encode(
        x_context, y_context, x_data
    )

    tuples_D = [(z_D_0, z_mu_D_0, z_std_D_0)]
    tuples_C = [(z_C_0, z_mu_C_0, z_std_C_0)]

    f_t_list: List[Tensor] = []
    b_t_list: List[Tensor] = []

    for t in range(1, diffu_vi_np_model.diffu_model.num_steps + 1):
        z_D_t, z_mu_D_t, z_std_D_t = diffu_vi_np_model.diffu_model.forward_transition(
            tuples_D[t - 1][0], torch.tensor([t - 1]).to(device)
        )

        z_C_t, z_mu_C_t, z_std_C_t = diffu_vi_np_model.diffu_model.forward_transition(
            tuples_C[t - 1][0], torch.tensor([t - 1]).to(device)
        )

        tuples_D.append((z_D_t, z_mu_D_t, z_std_D_t))
        tuples_C.append((z_C_t, z_mu_C_t, z_std_C_t))

        _, z_mu_D_t_minus_1, z_std_D_t_minus_1 = (
            diffu_vi_np_model.diffu_model.backward_transition(
                tuples_D[t][0], torch.tensor([t - 1]).to(device)
            )
        )

        b_t = Normal(z_mu_D_t_minus_1, z_std_D_t_minus_1).log_prob(tuples_D[t - 1][0]).mean(dim=0).sum()  # type: ignore
        f_t = Normal(tuples_D[t][1], tuples_D[t][2]).log_prob(tuples_D[t][0]).mean(dim=0).sum()  # type: ignore

        b_t_list.append(b_t)
        f_t_list.append(f_t)

    _, y_mu, y_std = diffu_vi_np_model.np_model.decode(x_data, r, tuples_D[-1][0])

    log_like = Normal(y_mu, y_std).log_prob(y_data).mean(dim=0).sum()  # type: ignore
    diffu_loss = (
        torch.stack(b_t_list).sum() - torch.stack(f_t_list).sum()
        if len(b_t_list) > 0 and len(f_t_list) > 0
        else torch.tensor(0.0).to(device)
    )
    flexible_prior = Normal(tuples_C[-1][1], tuples_C[-1][2]).log_prob(tuples_D[-1][0]).mean(dim=0).sum()  # type: ignore
    simple_prio = Normal(tuples_D[0][1], tuples_D[0][2]).log_prob(tuples_D[0][0]).mean(dim=0).sum()  # type: ignore

    loss = -(log_like + diffu_loss + flexible_prior - simple_prio)

    return loss, log_like, diffu_loss
