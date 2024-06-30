import os
import sys
import traceback
from torch.utils.data import DataLoader

sys.path.append("src/")

from dataloader import Loader
from helper import helpers
from utils import config, dump, load, device_init, weight_init, CustomException


class Trainer:
    def __init__(
        self,
        epochs=100,
        lr=0.0002,
        beta1=0.5,
        beta2=0.999,
        weight_decay=0.0001,
        momentum=0.9,
        device="cuda",
        adam=True,
        SGD=False,
        l1_regularization=False,
        l2_regularization=False,
        lr_scheduler=False,
        MLFlow=True,
        display=True,
        is_weight_init=True,
    ):
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = device
        self.adam = adam
        self.SGD = SGD
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.lr_scheduler = lr_scheduler
        self.MLFlow = MLFlow
        self.display = display
        self.is_weight_init = is_weight_init

        try:
            self.init = helpers(
                adam=self.adam,
                SGD=self.SGD,
                beta1=self.beta1,
                beta2=self.beta2,
                momentum=self.momentum,
                lr=self.lr,
            )

        except CustomException as e:
            print(e)
            traceback.print_exc()

        else:
            self.netG = self.init["netG"]
            self.netD = self.init["netD"]

            self.train_dataloader = self.init["train_dataloader"]
            self.valid_dataloader = self.init["valid_dataloader"]

            self.optimizerG = self.init["optimizerG"]
            self.optimizerD = self.init["optimizerD"]

            self.adversarial_loss = self.init["adversarial_loss"]
            self.pixelwise_loss = self.init["pixelwise_loss"]

            assert (
                self.init["train_dataloader"].__class__.__name__ == DataLoader.__name__
            )
            assert (
                self.init["valid_dataloader"].__class__.__name__ == DataLoader.__name__
            )

            assert self.init["netG"].__class__.__name__ == "Generator"
            assert self.init["netD"].__class__.__name__ == "Discriminator"

            assert self.init["adversarial_loss"].__class__.__name__ == "AdversarialLoss"
            assert self.init["pixelwise_loss"].__class__.__name__ == "PixelLoss"

            if self.is_weight_init:
                self.netG.apply(weight_init)
                self.netD.apply(weight_init)

            self.device = device_init(device=device)

            self.netG.to(self.device)
            self.netD.to(self.device)


if __name__ == "__main__":
    trainer = Trainer()
