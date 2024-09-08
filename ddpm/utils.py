from typing import List, Tuple

import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from ddpm.ddpm import DDPM


def train(
    device: torch.device,
    model: DDPM,
    dataloader: DataLoader,  # type: ignore
    optimizer: Optimizer,
    num_epochs: int = 50,
    num_timesteps: int = 1000,
) -> List[float]:

    criterion = nn.MSELoss()
    global_step = 0
    losses = []

    model.train()

    with torch.inference_mode(False):

        for epoch in range(num_epochs):

            progress_bar = tqdm(total=len(dataloader))
            progress_bar.set_description(f"Epoch {epoch}")

            for batch in dataloader:
                batch = batch[0].to(device)
                noise = torch.randn(batch.shape).to(device)
                timesteps = torch.randint(0, num_timesteps, (batch.shape[0],)).to(
                    device
                )

                noisy = model.add_noise(batch, noise, timesteps)
                noise_pred = model.reverse(noisy, timesteps)

                loss = criterion(noise_pred, noise)
                losses.append(loss.detach().item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": global_step}
                progress_bar.set_postfix(**logs)
                global_step += 1

            progress_bar.close()

    return losses


def show_images(images: List[Tensor]) -> None:
    images = [im.permute(1, 2, 0).numpy() for im in images]

    # Defining number of rows and columns
    fig = plt.figure(figsize=(3, 3))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx], cmap="gray")
                plt.axis("off")
                idx += 1

    # Showing the figure
    plt.show()


def generate_image(
    ddpm: DDPM, device: torch.device, sample_size: int, channel: int, size: int
) -> Tuple[List[Tensor], List[Tensor]]:
    """Generate the image from the Gaussian noise"""

    frames = []
    frames_mid = []

    ddpm.eval()
    with torch.no_grad():
        timesteps = list(range(ddpm.num_timesteps))[::-1]
        sample = torch.randn(sample_size, channel, size, size).to(device)

        for i, t in enumerate(tqdm(timesteps)):
            time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
            residual = ddpm.reverse(sample, time_tensor)
            sample = ddpm.step(residual, time_tensor[0], sample)

            if t == 500:
                for i in range(sample_size):
                    frames_mid.append(sample[i].detach().cpu())

        for i in range(sample_size):
            frames.append(sample[i].detach().cpu())
    return frames, frames_mid
