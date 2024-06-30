import os
import sys
import torch
import numpy as np
import torch.nn as nn
import traceback
from tqdm import tqdm
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

    def update_netG(self, **kwargs):
        self.optimizerG.zero_grad()

        X = kwargs["X"]
        y = kwargs["y"]

        generated_inpaint = self.netG(X)
        predicted_inpaint = self.netD(generated_inpaint)
        predicted_inpaint_loss = self.adversarial_loss(
            predicted_inpaint, torch.ones_like(predicted_inpaint)
        )

        pixelwise_loss = self.pixelwise_loss(generated_inpaint, y)

        total_netG_loss = 0.001 * predicted_inpaint_loss + 0.999 * pixelwise_loss

        total_netG_loss.backward()
        self.optimizerG.step()

        return total_netG_loss.item()

    def update_netD(self, **kwargs):
        self.optimizerD.zero_grad()

        X = kwargs["X"]
        y = kwargs["y"]

        generated_inpaint = self.netG(X)
        predicted_inpaint = self.netD(generated_inpaint)
        predicted_inpaint_loss = self.adversarial_loss(
            predicted_inpaint, torch.zeros_like(predicted_inpaint)
        )

        predicted_real = self.netD(y)
        predicted_real_loss = self.adversarial_loss(
            predicted_real, torch.ones_like(predicted_real)
        )

        total_netD_loss = 0.5 * (predicted_inpaint_loss + predicted_real_loss)

        total_netD_loss.backward()
        self.optimizerD.step()

        return total_netD_loss.item()

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            self.netG_loss = []
            self.netD_loss = []

            for index, (X, y) in enumerate(self.train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)

                self.netD_loss.append(self.update_netD(X=X, y=y))
                self.netG_loss.append(self.update_netG(X=X, y=y))

            print(np.mean(self.netD_loss), np.mean(self.netG_loss))


if __name__ == "__main__":
    trainer = Trainer(epochs=1, device="mps")
    trainer.train()
