import joblib
import yaml
import torch
import torch.nn as nn


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)

    else:
        raise ValueError("value and filename must be provided".capitalize())


def load(filename=None):
    if filename is not None:
        return joblib.load(filename=filename)

    else:
        raise ValueError("filename should be defined".capitalize())


def device_init(device="cuda"):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    else:
        return torch.device("cpu")


def weight_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)


class CustomException(Exception):
    def __init__(self, message=None):
        if message is not None:
            return message
