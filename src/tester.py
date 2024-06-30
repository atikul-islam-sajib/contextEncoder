import os
import sys
import torch
import argparse
import traceback
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append("src/")

from generator import Generator
from utils import config, load, device_init, CustomException


class Tester:
    def __init__(self, model="best", device="cuda", dataloader="valid"):
        self.model = model
        self.device = device
        self.dataloader = dataloader

        self.device = device_init(device=device)

    def select_model(self):
        if self.model == "best":
            best_model_path = config()["path"]["BEST_MODEL_PATH"]

            if os.path.exists(best_model_path):
                model = os.path.join(best_model_path, "best_model.pth")
                model = torch.load(model)
                model = model["netG"]

                return model

            else:
                raise CustomException("Cannot be found the best model".capitalize())

    def select_dataloader(self):
        processed_path = config()["path"]["PROCESSED_DATA_PATH"]

        if self.dataloader == "train":
            if os.path.exists(processed_path):
                train_dataloader = load(
                    filename=os.path.join(processed_path, "train_dataloader.pkl")
                )
                return train_dataloader

            else:
                raise CustomException("Cannot be found the processed data".capitalize())

        else:
            if os.path.exists(processed_path):
                valid_dataloader = load(
                    filename=os.path.join(processed_path, "test_dataloader.pkl")
                )
                return valid_dataloader

            else:
                raise CustomException("Cannot be found the processed data".capitalize())

    def plot(self):
        try:
            self.netG = Generator()
            self.netG.load_state_dict(self.select_model())
            self.netG.to(self.device)

        except CustomException as e:
            print(e)
            traceback.print_exc()

        except Exception as e:
            print(e)
            traceback.print_exc()

        try:
            dataloader = self.select_dataloader()

            assert dataloader.__class__.__name__ == DataLoader.__name__

        except CustomException as e:
            print(e)
            traceback.print_exc()

        except Exception as e:
            print(e)
            traceback.print_exc()

        else:
            data, label = next(iter(dataloader))
            predicted = self.netG(data.to(self.device))

        number_of_rows = data.size(0) // 2
        number_of_columns = data.size(0) // number_of_rows

        plt.figure(figsize=(20, 15))

        for index, image in enumerate(predicted):
            predicted_inpaint = image.permute(1, 2, 0).cpu().detach().numpy()
            predicted_inpaint = (predicted_inpaint - predicted_inpaint.min()) / (
                predicted_inpaint.max() - predicted_inpaint.min()
            )

            X = data[index].permute(1, 2, 0).cpu().detach().numpy()
            X = (X - X.min()) / (X.max() - X.min())

            y = label[index].permute(1, 2, 0).cpu().detach().numpy()
            y = (y - y.min()) / (y.max() - y.min())

            plt.subplot(3 * number_of_rows, 3 * number_of_columns, 3 * index + 1)
            plt.imshow(X)
            plt.title("X")
            plt.axis("off")

            plt.subplot(3 * number_of_rows, 3 * number_of_columns, 3 * index + 2)
            plt.imshow(y)
            plt.title("y")
            plt.axis("off")

            plt.subplot(3 * number_of_rows, 3 * number_of_columns, 3 * index + 3)
            plt.imshow(predicted_inpaint)
            plt.title("pred")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(config()["path"]["SAVE_TEST_IMAGE_PATH"], "result.png")
        )
        plt.show()

        print(
            "Result image is saved in {}".format(
                os.path.join(config()["path"]["SAVE_TEST_IMAGE_PATH"], "result.png")
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tester code for Context Encoder".title()
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config()["tester"]["model"],
        help="Define the model for further analysis".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["tester"]["device"],
        help="Define the device for further analysis".capitalize(),
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default=config()["tester"]["dataloader"],
        help="Define the dataloader for further analysis".capitalize(),
    )

    args = parser.parse_args()

    test = Tester(model=args.model, device=args.device, dataloader=args.dataloader)

    test.plot()
