import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append("learning")
sys.path.append("model")

from dataset_loader import ChestXrayDataset
from model import ChestXrayModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


LABELS = [
    "Pneumonia",
    "Pneumothorax",
    "Cardiomegaly",
    "Effusion",
    "Atelectasis",
    "No Finding"
]


test_dataset = ChestXrayDataset(
    "dataset/test.csv",
    "dataset/images"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


model = ChestXrayModel()

model.load_state_dict(
    torch.load(
        "model/chest_xray_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = (torch.sigmoid(outputs) >= 0.5).float()

        correct += (predictions == labels).sum().item()
        total += labels.numel()


accuracy = 100 * correct / total


print("\n==============================")
print("MODEL TESTING COMPLETED")
print("==============================")
print("Test images:", len(test_dataset))
print("Accuracy:", round(accuracy, 2), "%")
print("==============================")