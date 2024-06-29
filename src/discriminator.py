import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

from discriminator_block import DiscriminatorBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels

        self.out_channels = 64
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1

        self.layers = []

        for idx in range(4):
            self.layers.append(
                DiscriminatorBlock(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    stride=1 if (idx + 1) == 4 else 2,
                    use_normalization=False if idx == 0 else True,
                )
            )
            self.in_channels = self.out_channels
            self.out_channels *= 2

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels // self.in_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride // self.stride,
                    padding=self.padding,
                )
            )
        )

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.model(x)

        else:
            raise ValueError("Input should be in the format of the tensor".capitalize())


if __name__ == "__main__":
    netD = Discriminator()

    print(netD)
