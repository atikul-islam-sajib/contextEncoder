import os
import sys
import argparse
import torch
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph

sys.path.append("src/")

from utils import config
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

    @staticmethod
    def total_params(model):
        if isinstance(model, nn.Module):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        else:
            raise ValueError("Input should be in the format of the module".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discrimiantor block for Context Encoder".title()
    )

    parser.add_argument(
        "--in_channels", type=int, default=3, help="Define the in_channels".capitalize()
    )
    parser.add_argument(
        "--netD",
        action="store_true",
        help="Define the discriminator network".capitalize(),
    )

    args = parser.parse_args()

    if args.netD:
        netD = Discriminator(in_channels=args.in_channels)

        assert netD(torch.randn(1, 3, 64, 64)).size() == (1, 1, 8, 8)

        print(summary(model=netD, input_size=(3, 64, 64)))

        draw_graph(
            model=netD, input_data=torch.randn(1, 3, 64, 64)
        ).visual_graph.render(
            filename=os.path.join(config()["path"]["ARTIFACTS_PATH"], "netD"),
            format="png",
        )

    else:
        raise Exception("Please check the model arguments".capitalize())
