import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels

        self.kernel_size = 3
        self.stride = 2
        self.padding = 1
