from typing import Tuple

import torch
from torch import Tensor


def img_to_sequence(img: Tensor) -> Tensor:
    # (batch_size, channels, height, width)

    indices = (
        torch.arange(img.shape[1] * img.shape[2] * img.shape[3])
        .view(1, -1, 1)
        .repeat(img.shape[0], 1, 1)
    )
    flattend = img.view(img.shape[0], -1, 1)
    context = torch.cat([indices, flattend], dim=2)
    # -> (batch_size, channels * height * width, 2)

    return context


# seq_length = channels * height * width


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


if __name__ == "__main__":
    img = torch.randn(4, 3, 2, 2)
    print(img)

    sequence = img_to_sequence(img)
    print(sequence)

    context, target = split_context_target(sequence, 0.5)
    print(context)
    print(target)
