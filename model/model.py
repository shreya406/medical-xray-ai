import torch
import torch.nn as nn
from torchvision import models


class ChestXrayModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            6
        )

    def forward(self, x):

        return self.model(x)


if __name__ == "__main__":

    model = ChestXrayModel()

    print(model)

    test_input = torch.randn(
        1,
        3,
        224,
        224
    )

    output = model(test_input)

    print("Output shape:", output.shape)