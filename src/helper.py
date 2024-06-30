import os
import sys
import torch
import traceback
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append("src/")

from utils import config, load, CustomException
from dataloader import Loader
from generator import Generator
from discriminator import Discriminator
from adversarial_loss import AdversarialLoss
from pixelwise_loss import PixelLoss


def load_dataloader():
    processed_path = config()["path"]["PROCESSED_DATA_PATH"]

    if os.path.exists(processed_path):
        train_dataloader = load(os.path.join(processed_path, "train_dataloader.pkl"))
        valid_dataloader = load(os.path.join(processed_path, "test_dataloader.pkl"))

        return {
            "train_dataloader": train_dataloader,
            "valid_dataloader": valid_dataloader,
        }
    else:
        raise CustomException("Cannot import the dataloader".capitalize())


def helpers(**kwargs):
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]
    beta1 = kwargs["beta1"]
    beta2 = kwargs["beta2"]
    momentum = kwargs["momentum"]
    lr = kwargs["lr"]

    try:
        netG = Generator(in_channels=3, out_channels=64)
    except Exception as e:
        print("An error is occured {}".format(e))
        traceback.print_exc()

    try:
        netD = Discriminator(in_channels=3)
    except Exception as e:
        print("An error is occured {}".format(e))
        traceback.print_exc()

    if adam:
        optimizerG = optim.Adam(params=netG.parameters(), lr=lr, betas=(beta1, beta2))
        optimizerD = optim.Adam(params=netD.parameters(), lr=lr, betas=(beta1, beta2))

    if SGD:
        optimizerG = optim.SGD(params=netG.parameters(), lr=lr, momentum=momentum)
        optimizerD = optim.SGD(params=netD.parameters(), lr=lr, momentum=momentum)

    adversarial_loss = AdversarialLoss(reduction="mean")
    pixelwise_loss = PixelLoss(reduction="mean")

    try:
        dataloader = load_dataloader()

    except CustomException as e:
        print("An error is occured {}".format(e))
        traceback.print_exc()

    except Exception as e:
        print("An error is occured {}".format(e))
        traceback.print_exc()

    return {
        "train_dataloader": dataloader["train_dataloader"],
        "valid_dataloader": dataloader["valid_dataloader"],
        "netG": netG,
        "netD": netD,
        "optimizerG": optimizerG,
        "optimizerD": optimizerD,
        "adversarial_loss": adversarial_loss,
        "pixelwise_loss": pixelwise_loss,
    }


if __name__ == "__main__":
    init = helpers(
        adam=True, SGD=False, beta1=0.5, beta2=0.999, momentum=0.9, lr=0.0002
    )

    assert init["train_dataloader"].__class__.__name__ == DataLoader.__name__
    assert init["valid_dataloader"].__class__.__name__ == DataLoader.__name__

    assert init["netG"].__class__.__name__ == "Generator"
    assert init["netD"].__class__.__name__ == "Discriminator"

    assert init["optimizerG"].__class__.__name__ == "Adam"
    assert init["optimizerD"].__class__.__name__ == "Adam"

    assert init["adversarial_loss"].__class__.__name__ == "AdversarialLoss"
    assert init["pixelwise_loss"].__class__.__name__ == "PixelLoss"
