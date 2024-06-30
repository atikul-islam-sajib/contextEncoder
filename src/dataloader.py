import os
import sys
import cv2
import zipfile
import argparse
import traceback
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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
            batch_size=self.batch_size * 8,
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

    @staticmethod
    def dataset_details():
        processed_path = config()["path"]["PROCESSED_DATA_PATH"]
        artifacts_path = config()["path"]["ARTIFACTS_PATH"]

        if os.path.exists(processed_path):
            train_dataloader = load(
                filename=os.path.join(processed_path, "train_dataloader.pkl")
            )
            valid_dataloader = load(
                filename=os.path.join(processed_path, "test_dataloader.pkl")
            )

            train_data, train_label = next(iter(train_dataloader))
            valid_data, valid_label = next(iter(valid_dataloader))

            pd.DataFrame(
                {
                    "total_data_points": (sum(X.size(0) for X, _ in train_dataloader))
                    + sum(X.size(0) for X, _ in valid_dataloader),
                    "train_data_points": sum(X.size(0) for X, _ in train_dataloader),
                    "valid_data_points": sum(X.size(0) for X, _ in valid_dataloader),
                    "train_image_size(X)": str(train_data.size()),
                    "valid_image_size(X)": str(valid_data.size()),
                    "train_image_size(y)": str(train_label.size()),
                    "valid_image_size(y)": str(valid_label.size()),
                },
                index=["Quantity"],
            ).to_csv(os.path.join(artifacts_path, "dataset_details.csv"))

    @staticmethod
    def plot_images():
        processed_path = config()["path"]["PROCESSED_DATA_PATH"]
        artifacts_path = config()["path"]["ARTIFACTS_PATH"]

        train_dataloader = load(os.path.join(processed_path, "test_dataloader.pkl"))
        data, label = next(iter(train_dataloader))

        number_of_rows = data.size(0) // 2
        number_of_columns = data.size(0) // number_of_rows

        plt.figure(figsize=(20, 10))

        for index, image in enumerate(data):
            X = image.permute(1, 2, 0).detach().numpy()
            X = (X - X.min()) / (X.max() - X.min())

            y = label[index].permute(1, 2, 0).detach().numpy()
            y = (y - y.min()) / (y.max() - y.min())

            plt.subplot(2 * number_of_rows, 2 * number_of_columns, 2 * index + 1)
            plt.imshow(X)
            plt.title("X")
            plt.axis("off")

            plt.subplot(2 * number_of_rows, 2 * number_of_columns, 2 * index + 2)
            plt.imshow(y)
            plt.title("y")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_path, "images.png"))
        plt.show()

        print("Image is saved in the folder of {}".format(artifacts_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataloader class for context Loader".title()
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=config()["dataloader"]["image_path"],
        help="Define the dataset".capitalize(),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=config()["dataloader"]["channels"],
        help="Define the number of channels".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config()["dataloader"]["batch_size"],
        help="Define the batch size".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=config()["dataloader"]["split_size"],
        help="Define the split size".capitalize(),
    )
    args = parser.parse_args()

    loader = Loader(
        image_path=args.image_path,
        channels=args.channels,
        batch_size=args.batch_size,
        split_size=args.split_size,
    )

    loader.unzip_folder()
    loader.feature_extractor()
    loader.create_dataloader()

    Loader.dataset_details()
    Loader.plot_images()
