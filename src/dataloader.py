import os
import sys
import cv2
import zipfile
import traceback
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

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

        self.X = []
        self.y = []

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

    def split_dataset(self, X, y):
        if isinstance(X, list) and isinstance(y, list):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.split_size, random_state=42, shuffle=True
            )

            return X_train, X_test, y_train, y_test

        else:
            raise ValueError("X and y should be in the list format".capitalize())

    def feature_extractor(self):
        self.processed_data_path = self.CONFIG["path"]["PROCESSED_DATA_PATH"]

        X = os.path.join(self.processed_data_path, "dataset", "X")
        y = os.path.join(self.processed_data_path, "dataset", "y")

        for image in tqdm(os.listdir(X)):
            if image in os.listdir(y):
                image_X = os.path.join(X, image)
                image_Y = os.path.join(y, image)

                if (image_X is not None) and (image_Y is not None):
                    image_X = cv2.imread(image_X)
                    image_Y = cv2.imread(image_Y)

                    image_X = cv2.cvtColor(image_X, cv2.COLOR_BGR2RGB)
                    image_Y = cv2.cvtColor(image_Y, cv2.COLOR_BGR2RGB)

                    image_X = Image.fromarray(image_X)
                    image_Y = Image.fromarray(image_Y)

                    image_X = self.transforms(type="train")(image_X)
                    image_Y = self.transforms(type="valid")(image_Y)

                    self.X.append(image_X)
                    self.y.append(image_Y)

                else:
                    raise Exception(
                        "Image {} is not found in the path of {}".format(
                            image, self.processed_data_path
                        )
                    )

        try:
            X_train, X_test, y_train, y_test = self.split_dataset(X=self.X, y=self.y)

        except ValueError as e:
            print(e)
            traceback.print_exc()

        except Exception as e:
            print(e)
            traceback.print_exc()

        else:
            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

    def create_dataloader(self):
        dataset = self.feature_extractor()

        train_dataloader = DataLoader(
            dataset=list(zip(dataset["X_train"], dataset["y_train"])),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_dataloader = DataLoader(
            dataset=list(zip(dataset["X_test"], dataset["y_test"])),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for value, filename in [
            (train_dataloader, "train_dataloader"),
            (test_dataloader, "test_dataloader"),
        ]:
            dump(
                value=value,
                filename=os.path.join(
                    config()["path"]["PROCESSED_DATA_PATH"], filename + ".pkl"
                ),
            )

        print(
            "data is stored in the folder {}".format(
                config()["path"]["PROCESSED_DATA_PATH"]
            )
        )


if __name__ == "__main__":
    loader = Loader(
        image_path="./data/raw/dataset.zip",
        channels=3,
        batch_size=1,
        split_size=0.25,
    )

    # loader.unzip_folder()
    # loader.feature_extractor()
    loader.create_dataloader()
