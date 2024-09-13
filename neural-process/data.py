import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SineDataset(Dataset):  # type: ignore
    def __init__(
        self,
        amplitude_interval: Tuple[float, float] = (-1.0, 1.0),
        shift_interval: Tuple[float, float] = (-0.5 * math.pi, 0.5 * math.pi),
        num_samples: int = 10000,
        num_points: int = 100,
    ) -> None:
        self.amplitude_interval = amplitude_interval
        self.shift_interval = shift_interval
        self.num_samples = num_samples
        self.num_points = num_points

        # Generate data
        self.data = []

        a_min, a_max = self.amplitude_interval
        b_min, b_max = self.shift_interval

        for _ in range(num_samples):
            a = (a_max - a_min) * np.random.rand()  # + a_min
            b = (b_max - b_min) * np.random.rand()  # + b_min

            x = torch.linspace(-math.pi, math.pi, num_points)
            y = a * torch.sin(x - b)

            self.data.append((x, y))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return self.num_samples
