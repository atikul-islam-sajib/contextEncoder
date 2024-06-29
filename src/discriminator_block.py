import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")


class DiscriminatorBlock(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=64,
        momentum=0.8,
        use_normalization=True,
        stride=2,
    ):
        super(DiscriminatorBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = 3
        self.stride = stride
        self.padding = 1
        self.slope = 0.2

        self.momentum = momentum

        self.is_normalization = use_normalization

        self.discriminator_block = self.block()

    def block(self):
        self.layers = []

        self.layers.append(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        )

        if self.is_normalization:
            self.layers.append(
                nn.InstanceNorm2d(
                    num_features=self.out_channels, momentum=self.momentum
                )
            )

        self.layers.append(nn.LeakyReLU(negative_slope=self.slope, inplace=True))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.model(x)

        else:
            raise ValueError("Input should be in the format of the tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discriminator Block for the Context Encoder".title()
    )
    parser.add_argument("--in_channels", type=int, default=3, help="Input channels")
    parser.add_argument("--out_channels", type=int, default=64, help="Output channels")
    parser.add_argument(
        "--kernel_size", type=int, default=3, help="Kernel size for the convolution"
    )
    parser.add_argument(
        "--stride", type=int, default=2, help="Stride for the convolution"
    )
    parser.add_argument(
        "--padding", type=int, default=1, help="Padding for the convolution"
    )

    args = parser.parse_args()

    layers = []

    in_channels = args.in_channels
    out_channels = args.out_channels
    kernel_size = args.kernel_size
    stride = args.stride
    padding = args.padding

    for idx in range(4):
        layers.append(
            DiscriminatorBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1 if (idx + 1) == 4 else 2,
                use_normalization=False if idx == 0 else True,
            )
        )
        in_channels = out_channels
        out_channels *= 2

    layers.append(
        nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // in_channels,
                kernel_size=kernel_size,
                stride=stride // stride,
                padding=padding,
            )
        )
    )

    model = nn.Sequential(*layers)

    assert model(torch.randn(1, 3, 64, 64)).size() == (1, 1, 8, 8)
