import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "learning"
    )
)

from dataset_loader import ChestXrayDataset
from model import ChestXrayModel


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


train_dataset = ChestXrayDataset(
    "dataset/train.csv",
    "dataset/images"
)

validation_dataset = ChestXrayDataset(
    "dataset/validation.csv",
    "dataset/images"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


model = ChestXrayModel()
model = model.to(device)


criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


epochs = 5


for epoch in range(epochs):

    model.train()

    training_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        training_loss += loss.item()


    model.eval()

    validation_loss = 0.0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            validation_loss += loss.item()


    training_loss = (
        training_loss / len(train_loader)
    )

    validation_loss = (
        validation_loss / len(validation_loader)
    )


    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Training Loss: {training_loss:.4f} "
        f"- Validation Loss: {validation_loss:.4f}"
    )


torch.save(
    model.state_dict(),
    "model/chest_xray_model.pth"
)


print("\nTraining completed!")
print("Model saved to: model/chest_xray_model.pth")