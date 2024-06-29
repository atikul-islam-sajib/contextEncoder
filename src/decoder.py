import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels=4000, out_channels=512, kernel_size=4, stride=2, padding=1
    ):
        super(DecoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.decoder_block = self.block()

    def block(self):
        self.layers = []

        self.layers.append(
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        )

        self.layers.append(nn.BatchNorm2d(num_features=self.out_channels))
        self.layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*self.layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.decoder_block(x)

        else:
            raise ValueError("Input should be in the format of the tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decoder block for the ContextEncoder".title()
    )
    parser.add_argument(
        "--in_channels", type=int, default=4000, help="Input channels".capitalize()
    )
    parser.add_argument(
        "--out_channels", type=int, default=512, help="Output channels".capitalize()
    )
    parser.add_argument(
        "--kernel_size", type=int, default=4, help="Kernel size".capitalize()
    )
    parser.add_argument("--stride", type=int, default=2, help="Stride".capitalize())
    parser.add_argument("--padding", type=int, default=1, help="Padding".capitalize())

    layers = []

    args = parser.parse_args()

    in_channels = args.in_channels
    out_channels = args.out_channels
    kernel_size = args.kernel_size
    stride = args.stride
    padding = args.padding

    for _ in range(4):
        layers.append(
            DecoderBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        in_channels = out_channels
        out_channels = in_channels // 2

    layers.append(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=3,
            kernel_size=kernel_size - 1,
            stride=stride // stride,
            padding=padding,
        )
    )

    model = nn.Sequential(*layers)

    assert model(torch.randn(1, 4000, 4, 4)).size() == (1, 3, 64, 64)
