import os
import sys
import torch
import torch.nn as nn

sys.path.append("src/")

from encoder import EncoderBlock
from decoder import DecoderBlock


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = 4
        self.stride = 2
        self.padding = 1

        self.layers = []

        for _ in range(2):
            self.layers.append(
                EncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    use_norm=False,
                )
            )

            in_channels = out_channels

        for _ in range(3):
            self.layers.append(
                EncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels * 2,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    use_norm=True,
                )
            )

            in_channels = out_channels * 2
            out_channels = in_channels

        self.layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=4000,
                kernel_size=self.kernel_size // self.kernel_size,
                stride=self.stride // self.stride,
                padding=0,
            )
        )

        in_channels = 4000

        for _ in range(4):
            self.layers.append(
                DecoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                )
            )
            in_channels = out_channels
            out_channels = in_channels // 2

        self.layers.append(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=3,
                kernel_size=self.kernel_size - 1,
                stride=self.stride // self.stride,
                padding=self.padding,
            )
        )

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.model(x)
        else:
            raise ValueError("Input should be in the format of the tensor".capitalize())


if __name__ == "__main__":
    netG = Generator()

    print(netG(torch.randn(1, 3, 128, 128)).size())
