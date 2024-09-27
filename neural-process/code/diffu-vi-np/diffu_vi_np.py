from diffusion import DiffusionVariant
from neural_process import NeuralProcess
from torch import nn


class DiffusionNeuralProcess(nn.Module):
    def __init__(self, np_model: NeuralProcess, diffu_model: DiffusionVariant) -> None:
        super(DiffusionNeuralProcess, self).__init__()

        self.np_model = np_model
        self.diffu_model = diffu_model

    def forward(self) -> None:
        pass
