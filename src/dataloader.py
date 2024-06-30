import os
import sys
import zipfile
import torch
from PIL import Image
from torchvision import transforms

sys.path.append("src/")

from utils import config, dump, load


class Loader:
    def __init__(
        self, image_path=None, channels=3, image_size=128, batch_size=1, split_size=0.25
    ):
        self.image_path = image_path
        self.channels = channels
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size

        self.CONFIG = config()

    def unzip_folder(self):
        if not os.path.exists(self.CONFIG["path"]["PROCESSED_DATA_PATH"]):
            os.makedirs(self.CONFIG["path"]["PROCESSED_DATA_PATH"])

            print(
                "Folder is created successfully in the path of {}".format(
                    self.CONFIG["path"]["PROCESSED_DATA_PATH"]
                )
            )

        with zipfile.ZipFile(self.image_path, "r") as file:
            file.extractall(path=self.CONFIG["path"]["PROCESSED_DATA_PATH"])

        print(
            "Data unzipped successfully and store in the folder of {}".format(
                self.CONFIG["path"]["PROCESSED_DATA_PATH"]
            )
        )

    def transforms(self, type="train"):
        if type == "train":
            return transforms.Compose(
                [
                    transforms.Resize(
                        (self.image_size, self.image_size), Image.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.CenterCrop((self.image_size, self.image_size)),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        elif type == "valid":
            return transforms.Compose(
                [
                    transforms.Resize(
                        (self.image_size // 2, self.image_size // 2), Image.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.CenterCrop((self.image_size // 2, self.image_size // 2)),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )


if __name__ == "__main__":
    loader = Loader(
        image_path="./data/raw/dataset.zip",
        channels=3,
        batch_size=1,
        split_size=0.25,
    )

    loader.unzip_folder()
