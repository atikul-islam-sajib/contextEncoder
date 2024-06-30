import sys
import argparse

sys.path.append("src/")

from utils import config

from dataloader import Loader
from generator import Generator
from discriminator import Discriminator
from trainer import Trainer


def cli():
    parser = argparse.ArgumentParser(description="CLI for Context Encoder".title())
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
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )

    args = parser.parse_args()

    if args.train:

        loader = Loader(
            image_path=args.image_path,
            channels=args.channels,
            batch_size=args.batch_size,
            split_size=args.split_size,
        )

        # loader.unzip_folder()
        loader.feature_extractor()
        loader.create_dataloader()

        Loader.dataset_details()
        Loader.plot_images()

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


if __name__ == "__main__":
    cli()
