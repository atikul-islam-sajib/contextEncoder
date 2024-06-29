import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=64,
        kernel_size=4,
        stride=2,
        padding=1,
        use_norm=True,
    ):
        super(EncoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.is_normalization = use_norm

        self.encoder_block = self.block()

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
            self.layers.append(nn.BatchNorm2d(num_features=self.out_channels))

        self.layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        return nn.Sequential(*self.layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.encoder_block(x)
        else:
            raise ValueError("Input should be in the format of the tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder block for the ContextEncoder".title()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Define the in channels of the encoder model".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=64,
        help="Define the out channels of the encoder model".capitalize(),
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=4,
        help="Define the kernel size of the encoder model".capitalize(),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Define the stride of the encoder model".capitalize(),
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=1,
        help="define the padding of the encoder model".capitalize(),
    )

    args = parser.parse_args()

    in_channels = args.in_channels
    out_channels = args.out_channels
    kernel_size = args.kernel_size
    stride = args.stride
    padding = args.padding

    layers = []

    for _ in range(2):
        layers.append(
            EncoderBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_norm=False,
            )
        )

        in_channels = out_channels

    for _ in range(3):
        layers.append(
            EncoderBlock(
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_norm=True,
            )
        )

        in_channels = out_channels * 2
        out_channels = in_channels

    layers.append(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=4000,
            kernel_size=kernel_size // kernel_size,
            stride=stride // stride,
            padding=0,
        )
    )

    model = nn.Sequential(*layers)

    assert model(torch.randn(1, 3, 128, 128)).size() == (1, 4000, 4, 4)
