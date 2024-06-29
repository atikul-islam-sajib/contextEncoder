import torch
import torch.nn as nn


class PixelLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(PixelLoss, self).__init__()

        self.reduction = reduction

    def forward(self, pred, actual):
        if isinstance(pred, torch.Tensor) and isinstance(actual, torch.Tensor):
            self.loss = nn.L1Loss(reduction=self.reduction)

            return self.loss(pred, actual)

        else:
            raise ValueError(
                "Pred and actual should be in the tensor format".capitalize()
            )


if __name__ == "__main__":
    loss = PixelLoss()

    predicted = torch.tensor([1.0, 0.0, 1.0, 0.0])
    actual = torch.tensor([1.0, 0.0, 1.0, 0.0])

    assert loss(predicted, actual) == (0.0)
