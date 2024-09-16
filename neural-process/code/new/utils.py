from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from metalearning_benchmarks import MetaLearningBenchmark  # type: ignore
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


class MetaLearningDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(
        self,
        benchmark: MetaLearningBenchmark,
    ) -> None:
        self.benchmark = benchmark

    def __len__(self) -> int:
        return self.benchmark.n_task  # type: ignore

    def __getitem__(self, task_idx: int) -> Tuple[Tensor, Tensor]:
        task = self.benchmark.get_task_by_index(task_index=task_idx)
        return Tensor(task.x), Tensor(task.y)


class DataModule:
    def __init__(
        self,
        batch_size: int,
        dataset_and_split: Optional[
            Tuple[Dataset[Tuple[Tensor, Tensor]], Tuple[float, float]]
        ] = None,
        train_and_val_set: Optional[
            Tuple[Dataset[Tuple[Tensor, Tensor]], Dataset[Tuple[Tensor, Tensor]]]
        ] = None,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.train_set: Dataset[Tuple[Tensor, Tensor]]
        self.val_set: Dataset[Tuple[Tensor, Tensor]]

        if dataset_and_split is not None:

            self.dataset, split = dataset_and_split
            self.train_set, self.val_set = random_split(self.dataset, split)

        elif train_and_val_set is not None:
            self.train_set, self.val_set = train_and_val_set
        else:
            raise ValueError("Invalid dataset or split.")

    def train_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=15,
        )

    def val_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=15,
        )


def split_context_target(
    x_data: Tensor, y_data: Tensor, factor: float, random: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # (batch_size, seq_length, x_dim), (batch_size, seq_length, y_dim)

    assert x_data.shape[1] == y_data.shape[1]
    assert factor > 0 and factor < 1

    context_len = int(x_data.shape[1] * factor)
    shuffled_indices = (
        torch.randperm(x_data.shape[1]) if random else torch.arange(x_data.shape[1])
    )
    context_indices = shuffled_indices[:context_len]
    target_indices = shuffled_indices[context_len:]

    x_context = x_data[:, context_indices, :]
    y_context = y_data[:, context_indices, :]

    x_target = x_data[:, target_indices, :]
    y_target = y_data[:, target_indices, :]

    return x_context, y_context, x_target, y_target


def img_to_x_y(batch: Tensor) -> Tuple[Tensor, Tensor]:
    # (batch_size, channels, height, width)

    batch_size, channels, height, width = batch.shape
    n_pixels = height * width

    y_coords, x_coords = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    coords = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
    # (n_pixels, 2)
    x_data = coords.unsqueeze(0).expand(batch_size, n_pixels, 2)
    # (batch_size, n_pixels, 2)

    y_data = batch.permute(0, 2, 3, 1).reshape(batch_size, n_pixels, channels)
    # (batch_size, n_pixels, channels)

    return x_data, y_data


def x_y_to_img(x_data: Tensor, y_data: Tensor) -> Tensor:
    # (batch_size, n_pixels, 2), (batch_size, n_pixels, channels)

    batch_size, n_pixels, channels = y_data.shape
    height, width = int(np.sqrt(n_pixels)), int(np.sqrt(n_pixels))
    img = torch.zeros(batch_size, channels, height, width)

    for i in range(batch_size):
        pixel_coords = x_data[i].long()
        x_indices = pixel_coords[:, 0].view(-1)
        y_indices = pixel_coords[:, 1].view(-1)

        flat_indices = y_indices * width + x_indices

        flattened_image = img[i].view(channels, -1)
        # (n_channels, height * width)

        flattened_image.scatter_(
            1, flat_indices.unsqueeze(0).expand(channels, -1), y_data[i].permute(1, 0)
        )

    return img


@dataclass
class Config:
    task: str
    n_task: int
    n_datapoints_per_task: int
    output_noise: float
    seed_task: int
    seed_x: int
    seed_noise: int
    x_dim: int
    y_dim: int
    r_dim: int
    z_dim: int
    h_dim: int
    split: Tuple[float, float]
    batch_size: int
    num_epochs: int
    learning_rate: float

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)
