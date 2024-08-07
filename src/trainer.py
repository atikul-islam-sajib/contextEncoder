import os
import sys
import torch
import mlflow
import argparse
import warnings
import traceback
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from dagshub import dagshub_logger
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

load_dotenv()

warnings.filterwarnings("ignore")

sys.path.append("src/")

from helper import helpers
from utils import config, dump, device_init, weight_init, CustomException


class Trainer:
    def __init__(
        self,
        epochs=100,
        lr=0.0002,
        beta1=0.5,
        beta2=0.999,
        weight_decay=0.0001,
        momentum=0.9,
        adversarial_lambda=0.001,
        pixelwise_lambda=0.999,
        steps=10,
        step_size=10,
        gamma=0.1,
        device="cuda",
        adam=True,
        SGD=False,
        l1_regularization=False,
        l2_regularization=False,
        elasticnet_regularization=False,
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
        self.elasticnet_regularization = elasticnet_regularization
        self.lr_scheduler = lr_scheduler
        self.MLFlow = MLFlow
        self.display = display
        self.is_weight_init = is_weight_init
        self.adversarial_lambda = adversarial_lambda
        self.pixelwise_lambda = pixelwise_lambda
        self.steps = steps
        self.step_size = step_size
        self.gamma = gamma

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

            if self.lr_scheduler:
                self.schedulerG = StepLR(
                    optimizer=self.optimizerG,
                    step_size=self.step_size,
                    gamma=self.gamma,
                )
                self.schedulerD = StepLR(
                    optimizer=self.optimizerD,
                    step_size=self.step_size,
                    gamma=self.gamma,
                )

            self.device = device_init(device=device)

            self.netG.to(self.device)
            self.netD.to(self.device)

            self.loss = float("inf")

            self.total_netG_loss = []
            self.total_netD_loss = []
            self.history = {"netG_loss": [], "netD_loss": []}

            os.getenv("MLFLOW_TRACKING_URI")
            os.getenv("MLFLOW_TRACKING_USERNAME")
            os.getenv("MLFLOW_TRACKING_PASSWORD")

            mlflow.set_experiment(experiment_name="Context Encoder based GAN".title())

    def l1_regularizer(self, model=None, value=0.01):
        if model is not None:
            return value * sum(torch.norm(params, 1) for params in model.parameters())
        else:
            raise CustomException(
                "Elastic net cannot be possible for regularization".capitalize()
            )

    def l2_regularizer(self, model=None, value=0.001):
        if model is not None:
            return value * sum(torch.norm(params, 2) for params in model.parameters())
        else:
            raise CustomException(
                "Elastic net cannot be possible for regularization".capitalize()
            )

    def elasticnet_regularizer(self, model=None, value=0.001):
        if model is not None:
            self.l1 = self.l1_regularization(model=model, value=value)
            self.l2 = self.l2_regularization(model=model, value=value)

            return value * (self.l1 + self.l2)

        else:
            raise CustomException(
                "Elastic net cannot be possible for regularization".capitalize()
            )

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

        total_netG_loss = (
            self.adversarial_lambda * predicted_inpaint_loss
            + self.pixelwise_lambda * pixelwise_loss
        )

        if self.l1_regularization:
            total_netG_loss += self.l1_regularizer(model=self.netG)

        if self.l2_regularization:
            total_netG_loss += self.l2_regularizer(model=self.netG)

        if self.elasticnet_regularization:
            total_netG_loss += self.elasticnet_regularizer(model=self.netG)

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

        if self.l1_regularization:
            total_netD_loss += self.l1_regularizer(model=self.netD)

        if self.l1_regularization:
            total_netD_loss += self.l2_regularizer(model=self.netD)

        if self.elasticnet_regularization:
            total_netD_loss += self.elasticnet_regularizer(model=self.netD)

        total_netD_loss.backward()
        self.optimizerD.step()

        return total_netD_loss.item()

    def show_progress(self, **kwargs):
        if self.display:
            print(
                "Epochs: [{}/{}] - netG_loss: [{:.4f}] - netD_loss: [{:.4f}]".format(
                    kwargs["epoch"],
                    self.epochs,
                    kwargs["netG_loss"],
                    kwargs["netD_loss"],
                )
            )
        else:
            print("Epochs: [{}/{}] is completed".format(kwargs["epoch"], self.epochs))

    def saved_checkpoints(self, **kwargs):
        train_models_path = config()["path"]["TRAIN_MODELS_PATH"]
        best_model_path = config()["path"]["BEST_MODEL_PATH"]

        netG_loss = kwargs["netG_loss"]
        epoch = kwargs["epoch"]

        if (not os.path.exists(train_models_path)) and (
            not os.path.exists(best_model_path)
        ):
            os.makedirs(train_models_path, exist_ok=True)
            os.makedirs(best_model_path, exist_ok=True)

        elif (os.path.exists(train_models_path)) and (os.path.exists(best_model_path)):
            if self.loss > netG_loss:
                self.loss = netG_loss

                torch.save(
                    {
                        "netG": self.netG.state_dict(),
                        "netG_loss": netG_loss,
                        "epoch": epoch,
                    },
                    os.path.join(best_model_path, "best_model.pth"),
                )

            torch.save(
                self.netG.state_dict(),
                os.path.join(train_models_path, "netG{}.pth".format(epoch)),
            )

        else:
            raise CustomException(
                "Cannot be saved the models in the checkpoints".capitalize()
            )

    def train(self):
        with mlflow.start_run(
            description="Context Encoders: Feature Learning by Inpainting Context Encoders for Inpainting & Self-Supervised Learning("
        ) as run:
            for epoch in tqdm(range(self.epochs)):
                self.netG_loss = []
                self.netD_loss = []

                for _, (X, y) in enumerate(self.train_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device)

                    self.netD_loss.append(self.update_netD(X=X, y=y))
                    self.netG_loss.append(self.update_netG(X=X, y=y))

                if self.lr_scheduler:
                    self.schedulerG.step()
                    self.schedulerD.step()

                self.show_progress(
                    netG_loss=np.mean(self.netG_loss),
                    netD_loss=np.mean(self.netD_loss),
                    epoch=epoch + 1,
                )

                try:
                    self.saved_checkpoints(
                        netG_loss=np.mean(self.netG_loss), epoch=epoch + 1
                    )
                except CustomException as e:
                    print(e)
                    traceback.print_exc()

                except Exception as e:
                    print(e)
                    traceback.print_exc()

                if (epoch + 1) % self.steps == 0:
                    X, y = next(iter(self.train_dataloader))
                    X = X.to(self.device)
                    y = y.to(self.device)

                    predicted_impaint = self.netG(X)

                    save_image(
                        predicted_impaint,
                        os.path.join(
                            config()["path"]["SAVE_IMAGE_PATH"],
                            "image{}.png".format(epoch + 1),
                        ),
                        nrow=1,
                    )

                self.history["netG_loss"].append(np.mean(self.netG_loss))
                self.history["netD_loss"].append(np.mean(self.netD_loss))

                mlflow.log_params(
                    {
                        "epochs": self.epochs,
                        "lr": self.lr,
                        "beta1": self.beta1,
                        "beta2": self.beta2,
                        "weight_decay": self.weight_decay,
                        "momentum": self.momentum,
                        "adversarial_lambda": self.adversarial_lambda,
                        "pixelwise_lamda": self.pixelwise_lambda,
                        "steps": self.steps,
                        "step_size": self.step_size,
                        "gamma": self.gamma,
                        "device": self.device,
                        "adam": self.adam,
                        "SGD": self.SGD,
                        "l1_regularization": self.l1_regularization,
                        "l2_regularization": self.l2_regularization,
                        "lr_scheduler": self.lr_scheduler,
                        "MLFlow": self.MLFlow,
                        "display": self.display,
                        "is_weight_init": self.is_weight_init,
                    }
                )

                mlflow.log_metric(
                    key="netG_loss", value=np.mean(self.netG_loss), step=epoch + 1
                )
                mlflow.log_metric(
                    key="netD_loss", value=np.mean(self.netD_loss), step=epoch + 1
                )

            dump(
                value=self.history,
                filename=os.path.join(
                    os.path.join(config()["path"]["METRCIS_PATH"], "history.pkl")
                ),
            )

            mlflow.pytorch.log_model(self.netG, "netG")
            mlflow.pytorch.log_model(self.netD, "netD")

            print(
                "Metrics in a pickle format is stored in the folder {}".format(
                    config()["path"]["METRCIS_PATH"]
                ).capitalize()
            )
            print("""mlflow is used, "run the command: mflow ui" """.capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer for Context Encoder".title())
    parser.add_argument(
        "--epochs",
        type=int,
        default=config()["trainer"]["epochs"],
        help="Define the epochs for training".capitalize(),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config()["trainer"]["lr"],
        help="Learning rate".capitalize(),
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=config()["trainer"]["beta1"],
        help="Beta1 for Adam optimizer".capitalize(),
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=config()["trainer"]["beta2"],
        help="Beta2 for Adam optimizer".capitalize(),
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config()["trainer"]["weight_decay"],
        help="Weight decay".capitalize(),
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=config()["trainer"]["momentum"],
        help="Momentum for SGD".capitalize(),
    )
    parser.add_argument(
        "--adversarial_lambda",
        type=float,
        default=config()["trainer"]["adversarial_lambda"],
        help="Adversarial lambda".capitalize(),
    )
    parser.add_argument(
        "--pixelwise_lambda",
        type=float,
        default=config()["trainer"]["pixelwise_lambda"],
        help="Pixelwise lambda".capitalize(),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=config()["trainer"]["steps"],
        help="Number of steps".capitalize(),
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=config()["trainer"]["step_size"],
        help="Step size for learning rate scheduler".capitalize(),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=config()["trainer"]["gamma"],
        help="Gamma for learning rate scheduler".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["trainer"]["device"],
        help="Device to run training on".capitalize(),
    )
    parser.add_argument(
        "--adam",
        type=bool,
        default=config()["trainer"]["adam"],
        help="Use Adam optimizer".capitalize(),
    )
    parser.add_argument(
        "--SGD",
        type=bool,
        default=config()["trainer"]["SGD"],
        help="Use SGD optimizer".capitalize(),
    )
    parser.add_argument(
        "--l1_regularization",
        type=bool,
        default=config()["trainer"]["l1_regularization"],
        help="Use L1 regularization".capitalize(),
    )
    parser.add_argument(
        "--l2_regularization",
        type=bool,
        default=config()["trainer"]["l2_regularization"],
        help="Use L2 regularization".capitalize(),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=config()["trainer"]["lr_scheduler"],
        help="Use learning rate scheduler".capitalize(),
    )
    parser.add_argument(
        "--MLFlow",
        type=bool,
        default=config()["trainer"]["MLFlow"],
        help="Use MLFlow for logging".capitalize(),
    )
    parser.add_argument(
        "--display",
        type=bool,
        default=config()["trainer"]["display"],
        help="Display training progress".capitalize(),
    )
    parser.add_argument(
        "--is_weight_init",
        type=bool,
        default=config()["trainer"]["is_weight_init"],
        help="Initialize weights".capitalize(),
    )

    args = parser.parse_args()

    trainer = Trainer(
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        adversarial_lambda=args.adversarial_lambda,
        pixelwise_lambda=args.pixelwise_lambda,
        steps=args.steps,
        step_size=args.step_size,
        gamma=args.gamma,
        device=args.device,
        adam=args.adam,
        SGD=args.SGD,
        l1_regularization=args.l1_regularization,
        l2_regularization=args.l2_regularization,
        lr_scheduler=args.lr_scheduler,
        MLFlow=args.MLFlow,
        display=args.display,
        is_weight_init=args.is_weight_init,
    )

    trainer.train()
