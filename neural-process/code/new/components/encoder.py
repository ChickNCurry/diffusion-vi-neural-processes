from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from deterministic_encoder import DeterministicEncoder
from diffusion_process import DiffusionProcess
from latent_encoder import LatentEncoder, z_tuple
from torch import Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        deterministic_encoder: DeterministicEncoder | None,
        latent_encoder: LatentEncoder | DiffusionProcess | None,
    ) -> None:
        super(Encoder, self).__init__()

        assert deterministic_encoder is not None or latent_encoder is not None

        self.deterministic_encoder = deterministic_encoder
        self.latent_encoder = latent_encoder

    def forward(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor
    ) -> Tuple[Tensor, List[z_tuple]]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)
        # (batch_size, target_size, x_dim)

        r = None
        z_repeated = None
        z_tuples: List[z_tuple] = []

        if self.deterministic_encoder is not None:
            r = self.deterministic_encoder(x_context, y_context, x_target)
            # (batch_size, target_size, r_dim)

        if self.latent_encoder is not None:
            z_tuples = self.latent_encoder(x_context, y_context, x_target)
            # (batch_size, z_dim)

            z_repeated = z_tuples[-1].z.unsqueeze(1).repeat(1, x_target.shape[1], 1)
            # (batch_size, target_size, z_dim)

        outputs = [t for t in [r, z_repeated] if t is not None]
        output = torch.cat(outputs, dim=-1)
        # (batch_size, target_size, r_dim + z_dim)

        return output, z_tuples
