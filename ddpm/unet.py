from typing import List

import torch
from torch import Tensor, nn


def sinusoidal_embedding(n: int, d: int) -> Tensor:
    embedding = torch.tensor(
        [[i / 10000 ** (2 * j / d) for j in range(d)] for i in range(n)]
    )

    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding


class MyConv(nn.Module):
    def __init__(
        self,
        shape: List[int],
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super(MyConv, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.ln(x)
        x = self.conv(x)
        x = self.activation(x)
        return x


def MyTinyBlock(size: int, in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        MyConv([in_c, size, size], in_c, out_c),
        MyConv([out_c, size, size], out_c, out_c),
        MyConv([out_c, size, size], out_c, out_c),
    )


def MyTinyUp(size: int, in_c: int) -> nn.Sequential:
    return nn.Sequential(
        MyConv([in_c, size, size], in_c, in_c // 2),
        MyConv([in_c // 2, size, size], in_c // 2, in_c // 4),
        MyConv([in_c // 4, size, size], in_c // 4, in_c // 4),
    )


class MyTinyUNet(nn.Module):
    # Here is a network with 3 down and 3 up with the tiny block
    def __init__(
        self,
        in_c: int = 1,
        out_c: int = 1,
        size: int = 32,
        n_steps: int = 1000,
        time_emb_dim: int = 100,
    ) -> None:
        super(MyTinyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = MyTinyBlock(size, in_c, 10)
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)
        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = MyTinyBlock(size // 2, 10, 20)
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)
        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = MyTinyBlock(size // 4, 20, 40)
        self.down3 = nn.Conv2d(40, 40, 4, 2, 1)

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyConv([40, size // 8, size // 8], 40, 20),
            MyConv([20, size // 8, size // 8], 20, 20),
            MyConv([20, size // 8, size // 8], 20, 40),
        )

        # Second half
        self.up1 = nn.ConvTranspose2d(40, 40, 4, 2, 1)
        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = MyTinyUp(size // 4, 80)
        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = MyTinyUp(size // 2, 40)
        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = MyTinyBlock(size, 20, 10)
        self.conv_out = nn.Conv2d(10, out_c, 3, 1, 1)

    def forward(
        self, x: Tensor, t: Tensor
    ) -> Tensor:  # x is (bs, in_c, size, size) t is (bs)
        t = self.time_embed(t)
        n = len(x)

        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (bs, 10, size/2, size/2)
        out2 = self.b2(
            self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1)
        )  # (bs, 20, size/4, size/4)
        out3 = self.b3(
            self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1)
        )  # (bs, 40, size/8, size/8)

        out_mid = self.b_mid(
            self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1)
        )  # (bs, 40, size/8, size/8)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (bs, 80, size/8, size/8)
        out4 = self.b4(
            out4 + self.te4(t).reshape(n, -1, 1, 1)
        )  # (bs, 20, size/8, size/8)
        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (bs, 40, size/4, size/4)
        out5 = self.b5(
            out5 + self.te5(t).reshape(n, -1, 1, 1)
        )  # (bs, 10, size/2, size/2)
        out = torch.cat((out1, self.up3(out5)), dim=1)  # (bs, 20, size, size)
        out = self.b_out(
            out + self.te_out(t).reshape(n, -1, 1, 1)
        )  # (bs, 10, size, size)
        out = self.conv_out(out)  # (bs, out_c, size, size)
        return out

    def _make_te(self, dim_in: int, dim_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )
