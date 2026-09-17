import sys
import os
import torch
from PIL import Image
from torchvision import transforms

sys.path.append("model")

from model import ChestXrayModel


LABELS = [
    "Pneumonia",
    "Pneumothorax",
    "Cardiomegaly",
    "Effusion",
    "Atelectasis",
    "No Finding"
]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


model = ChestXrayModel()

model.load_state_dict(
    torch.load(
        "model/chest_xray_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


image_path = "dataset/images/00002360_002.png"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)


with torch.no_grad():

    output = model(image)

    probabilities = torch.sigmoid(output)[0]


print("\n==============================")
print("CHEST X-RAY PREDICTION")
print("==============================")


for label, probability in zip(LABELS, probabilities):

    percentage = probability.item() * 100

    print(
        f"{label}: {percentage:.2f}%"
    )


print("==============================")