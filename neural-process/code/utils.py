from typing import Tuple

import torch
from torch import Tensor


def img_batch_to_sequence(batch: Tensor) -> Tensor:
    # (batch_size, channels, height, width)

    indices = (
        torch.arange(batch.shape[1] * batch.shape[2] * batch.shape[3])
        .view(1, -1, 1)
        .repeat(batch.shape[0], 1, 1)
    )

    flattend = batch.view(batch.shape[0], -1, 1)
    sequence = torch.cat([indices, flattend], dim=2)
    # -> (batch_size, channels * height * width, 2)

    # seq_length = channels * height * width

    return sequence


def split_context_target(
    sequence: Tensor, context_factor: float
) -> Tuple[Tensor, Tensor]:
    # (batch_size, seq_length, 2)

    context_size = int(sequence.shape[1] * context_factor)

    shuffled_indices = torch.randperm(sequence.shape[1])

    context_indices = shuffled_indices[:context_size]
    target_indices = shuffled_indices[context_size:]

    context = sequence[:, context_indices, :]
    target = sequence[:, target_indices, :]

    return context, target
