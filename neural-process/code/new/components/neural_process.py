from typing import List, Tuple

from decoder import Decoder
from encoder import Encoder
from torch import Tensor, nn


class NeuralProcess(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
    ) -> None:
        super(NeuralProcess, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def encode(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor, Tensor]]]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)
        # (batch_size, target_size, x_dim)

        output, z_tuples = self.encoder(x_context, y_context, x_target)
        # -> (batch_size, target_size, r_dim + z_dim + x_dim), (batch_size, z_dim)

        return output, z_tuples

    def decode(self, x_target: Tensor, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # (batch_size, target_size, r_dim + z_dim + x_dim)
        # (batch_size, z_dim)

        y, y_mu, y_std = self.decoder(x_target, input)
        # -> (batch_size, target_size, y_dim)

        return y, y_mu, y_std

    def forward(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tuple[Tensor, Tensor, Tensor]]]:
        # (batch_size, context_size, x_dim)
        # (batch_size, context_size, y_dim)
        # (batch_size, target_size, x_dim)

        output, z_tuples = self.encode(x_context, y_context, x_target)
        # -> (batch_size, target_size, r_dim + z_dim + x_dim), (batch_size, z_dim)

        y, y_mu, y_std = self.decode(x_target, output)
        # -> (batch_size, target_size, y_dim)

        return y, y_mu, y_std, z_tuples

    def sample(
        self, x_context: Tensor, y_context: Tensor, x_target: Tensor, num_samples: int
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tuple[Tensor, Tensor, Tensor]]]:
        # (1, context_size, x_dim)
        # (1, context_size, y_dim)
        # (1, target_size, x_dim)

        x_context = x_context.repeat(num_samples, 1, 1)
        # -> (num_samples, context_size, x_dim)

        y_context = y_context.repeat(num_samples, 1, 1)
        # -> (num_samples, context_size, y_dim)

        x_target = x_target.repeat(num_samples, 1, 1)
        # -> (num_samples, target_size, x_dim)

        y, y_mu, y_std, z_tuples = self.forward(x_context, y_context, x_target)
        # -> (num_samples, target_size, y_dim), (num_samples, z_dim)

        return y, y_mu, y_std, z_tuples
