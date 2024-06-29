import sys
import torch
import torch.nn as nn
import unittest

sys.path.append("src/")

from encoder import EncoderBlock
from decoder import DecoderBlock
from generator import Generator
from discriminator import Discriminator
from helper import helpers


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.encoder = EncoderBlock()
        self.decoder = DecoderBlock()
        self.netG = Generator()
        self.netD = Discriminator()

    def test_encoder_block(self):
        in_channels = 3
        out_channels = 64
        kernel_size = 4
        stride = 2
        padding = 1

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

        self.assertEqual(
            model(torch.randn(1, 3, 128, 128)).size(), torch.Size([1, 4000, 4, 4])
        )

    def test_decoder_block(self):
        in_channels = 4000
        out_channels = 512
        kernel_size = 4
        stride = 2
        padding = 1

        layers = []

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

        self.assertEqual(
            model(torch.randn(1, 4000, 4, 4)).size(), torch.Size([1, 3, 64, 64])
        )

    def test_netG_size(self):
        self.assertEqual(
            self.netG(torch.randn(1, 3, 128, 128)).size(), torch.Size([1, 3, 64, 64])
        )

    def test_netD_size(self):
        self.assertEqual(
            self.netD(torch.randn(1, 3, 64, 64)).size(), torch.Size([1, 1, 8, 8])
        )


if __name__ == "__main__":
    unittest.main()
