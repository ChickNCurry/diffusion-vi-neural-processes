from typing import Callable, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Normal
from torch import Tensor


class ConditionalGaussian:
    def __init__(
        self,
        device: torch.device,
        std_fn: Callable[[int], Tensor],
        mean_fn: Optional[Callable[[int], Tensor]] = None,
        context_bounds: List[int] = [-4, 4],
    ):
        self.act_dim = 1
        self.c_dim = 1
        if mean_fn is None:
            mean_fn = lambda c: torch.zeros_like(c)
        self.mean_fn = mean_fn
        self.std_fn = std_fn
        self.device = device
        self.context_bounds = context_bounds

    def get_targets(self, x: Tensor, c: int) -> Tensor:
        dist = Normal(
            loc=self.mean_fn(c).to(self.device), scale=self.std_fn(c).to(self.device)
        )  # type: ignore
        return dist.log_prob(x + 1e-45)

    def get_probs(self, x, c):
        return torch.exp(self.get_targets(x, c))

    def sample(self, n_samples, c):
        dist = Normal(
            loc=self.mean_fn(c).to(self.device), scale=self.std_fn(c).to(self.device)
        )
        return dist.sample().to(self.device)

    def sample_contexts(self, n_samples):
        return (
            torch.FloatTensor(n_samples, 1)
            .uniform_(self.context_bounds[0], self.context_bounds[1])
            .to(self.device)
        )

    def __call__(self, x, c):
        return self.get_targets(x, c)

    def visualize(
        self, fig=None, x_range=[-8, 8], res=100, device="cpu", show=False, pause=0.001
    ):
        x_test = np.linspace(x_range[0], x_range[1], res)
        c_test = np.linspace(self.context_bounds[0], self.context_bounds[1], res)
        x, c = np.meshgrid(x_test, c_test)
        x_flat = (
            torch.from_numpy(x.reshape(-1, 1).astype(np.float32)).detach().to(device)
        )
        c_flat = (
            torch.from_numpy(c.reshape(-1, 1).astype(np.float32)).detach().to(device)
        )
        contours = (self.get_probs(x_flat, c_flat).view(x.shape).detach() + 1e-6).to(
            "cpu"
        )
        if fig is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure(fig.number)
            if len(fig.axes) == 0:
                ax = plt.gca()
                fig.axes.append(ax)
            n = 1 if len(fig.axes) == 2 else 0
            ax = fig.axes[n]
        plt.ylim(*x_range)
        plt.xlim(*self.context_bounds)
        plt.xlabel("c")
        plt.contourf(c, x, contours, levels=100)
        plt.plot(
            c_test,
            mean_func(torch.tensor(c_test)) + 2 * std_func(torch.tensor(c_test)),
            c="w",
        )
        plt.plot(
            c_test,
            mean_func(torch.tensor(c_test)) - 2 * std_func(torch.tensor(c_test)),
            c="w",
        )
        if show:
            plt.show()
            plt.pause(pause)
        return fig

    def plot_samples(self, x, c, title=None, fig=None, show=False, pause=0.001):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure(fig.number)
            if len(fig.axes) == 0:
                ax = plt.gca()
                fig.axes.append(ax)
            n = 1 if len(fig.axes) == 2 else 0
            ax = fig.axes[n]
        plt.scatter(c, x, c="r")
        if title is not None:
            plt.title(title)
        if show:
            plt.show()
            plt.pause(pause)
        return fig


def std_func(c):
    return torch.ones_like(c) * 2
    # return c + 0.01
    # return torch.sin(5 * c) + 1.1


def mean_func(c):
    # return 3 * c + 1
    return_tensor = torch.sin(5 * c) + 1.1 + 2 * c - 1
    return_tensor[c > 1] += -6
    return return_tensor
    # return torch.sin(5 * c) + 1.1 + 2 * c - 1
    # return torch.zeros_like(c)


if __name__ == "__main__":
    env = ConditionalGaussian(std_func, mean_fn=mean_func, context_bounds=[0.001, 4])
    env.visualize(show=True)
    env.sample_contexts(100)
