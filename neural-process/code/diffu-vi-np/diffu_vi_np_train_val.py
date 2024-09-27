from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import wandb
from diffu_vi_np import DiffusionNeuralProcess
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
) -> Tuple[
    List[float], List[float], List[float], List[float], List[float], List[float]
]:

    # torch.autograd.set_detect_anomaly(True)

    avg_train_neg_log_likes = []
    avg_train_kl_divs = []
    avg_train_diffu_losses = []

    avg_val_neg_log_likes = []
    avg_val_kl_divs = []
    avg_val_diffu_losses = []

    for epoch in range(num_epochs):

        diffu_vi_np_model.train()
        with torch.inference_mode(False):

            train_neg_log_likes = []
            train_kl_divs = []
            train_diffu_losses = []

            dataloader = data_module.train_dataloader()
            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss, neg_log_like, kl_div, diffu_loss = step(
                    diffu_vi_np_model, device, batch, preprocessing
                )

                optimizer.zero_grad()
                loss.backward()  # type: ignore
                optimizer.step()

                loop.set_postfix(
                    epoch=epoch,
                    loss=loss.item(),
                    recon_loss=neg_log_like.item(),
                    kl_div=kl_div.item(),
                    diffu_loss=diffu_loss.item(),
                )

                train_neg_log_likes.append(neg_log_like.item())
                train_kl_divs.append(kl_div.item())
                train_diffu_losses.append(diffu_loss.item())

                if wandb_logging:
                    wandb.log(
                        {
                            "train/neg_log_like": neg_log_like.item(),
                            "train/kl_div": kl_div.item(),
                            "train/diffu_loss": diffu_loss.item(),
                        }
                    )

            avg_train_neg_log_like = float(np.mean(train_neg_log_likes))
            avg_train_kl_div = float(np.mean(train_kl_divs))
            avg_train_diffu_loss = float(np.mean(train_diffu_losses))

            avg_train_neg_log_likes.append(avg_train_neg_log_like)
            avg_train_kl_divs.append(avg_train_kl_div)
            avg_train_diffu_losses.append(avg_train_diffu_loss)

        diffu_vi_np_model.eval()
        with torch.no_grad():

            val_neg_log_likes = []
            val_kl_divs = []
            val_diffu_losses = []

            dataloader = data_module.val_dataloader()
            loop = tqdm(dataloader, total=len(dataloader))

            for batch in loop:

                loss, neg_log_like, kl_div, diffu_loss = step(
                    diffu_vi_np_model, device, batch, preprocessing
                )

                val_neg_log_likes.append(neg_log_like.item())
                val_kl_divs.append(kl_div.item())
                val_diffu_losses.append(diffu_loss.item())

                loop.set_postfix(
                    epoch=epoch,
                    loss=loss.item(),
                    recon_loss=neg_log_like.item(),
                    kl_div=kl_div.item(),
                    diffu_loss=diffu_loss.item(),
                )

                if wandb_logging:
                    wandb.log(
                        {
                            "val/neg_log_like": neg_log_like.item(),
                            "val/kl_div": kl_div.item(),
                            "val/diffu_loss": diffu_loss.item(),
                        }
                    )

            avg_val_neg_log_like = float(np.mean(val_neg_log_likes))
            avg_val_kl_div = float(np.mean(val_kl_divs))
            avg_val_diffu_loss = float(np.mean(val_diffu_losses))

            avg_val_neg_log_likes.append(avg_val_neg_log_like)
            avg_val_kl_divs.append(avg_val_kl_div)
            avg_val_diffu_losses.append(avg_val_diffu_loss)

    return (
        avg_train_neg_log_likes,
        avg_train_kl_divs,
        avg_train_diffu_losses,
        avg_val_neg_log_likes,
        avg_val_kl_divs,
        avg_val_diffu_losses,
    )


def step(
    diffu_vi_np_model: DiffusionNeuralProcess,
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

    r, z_0, z_0_mu_D, z_0_std_D = diffu_vi_np_model.np_model.encode(
        x_data, y_data, x_data
    )
    _, _, z_0_mu_C, z_0_std_C = diffu_vi_np_model.np_model.encode(
        x_context, y_context, x_data
    )

    z_t_list = [z_0]
    z_t_mu_list = [z_0_mu_D]
    z_t_std_list = [z_0_std_D]

    for t in range(1, diffu_vi_np_model.diffu_model.num_steps):
        z_t, z_t_mu, z_t_std = diffu_vi_np_model.diffu_model.forward_transition(
            z_t_list[t - 1], torch.tensor([t - 1]).to(device)
        )

        z_t_list.append(z_t)
        z_t_mu_list.append(z_t_mu)
        z_t_std_list.append(z_t_std)

    f_t_list: List[Tensor] = []
    b_t_list: List[Tensor] = []

    for t in range(1, diffu_vi_np_model.diffu_model.num_steps):
        f_t = Normal(z_t_mu_list[t - 1], z_t_std_list[t - 1]).log_prob(z_t_list[t]).mean(dim=0).sum()  # type: ignore

        _, b_t_mu, b_t_std = diffu_vi_np_model.diffu_model.backward_transition(
            z_t_list[t], torch.tensor([t]).to(device)
        )

        b_t = Normal(b_t_mu, b_t_std).log_prob(z_t_list[t - 1]).mean(dim=0).sum()  # type: ignore

        f_t_list.append(f_t)
        b_t_list.append(b_t)

    _, y_mu, y_std = diffu_vi_np_model.np_model.decode(x_data, r, z_t_list[-1])

    neg_log_like = -Normal(y_mu, y_std).log_prob(y_data).mean(dim=0).sum()  # type: ignore
    kl_div = kl_divergence(Normal(z_0_mu_D, z_0_std_D), Normal(z_0_mu_C, z_0_std_C)).mean(dim=0).sum()  # type: ignore
    diffu_loss = torch.stack(b_t_list).sum() - torch.stack(f_t_list).sum()

    loss = neg_log_like + kl_div - diffu_loss

    return loss, neg_log_like, kl_div, diffu_loss
