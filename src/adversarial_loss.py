import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(AdversarialLoss, self).__init__()

        self.reduction = reduction

    def forward(self, pred, actual):
        if (isinstance(pred, torch.Tensor)) and (isinstance(actual, torch.Tensor)):
            self.loss = nn.MSELoss(reduction=self.reduction)

            return self.loss(predicted, actual)

        else:
            raise ValueError(
                "Pred and actual should be in the tensor format".capitalize()
            )


if __name__ == "__main__":

    loss = AdversarialLoss()

    predicted = torch.tensor([1.0, 0.0, 1.0, 0.0])
    actual = torch.tensor([1.0, 0.0, 1.0, 0.0])

    assert loss(predicted, actual) == (0.0)
