import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


LABELS = [
    "Pneumonia",
    "Pneumothorax",
    "Cardiomegaly",
    "Effusion",
    "Atelectasis",
    "No Finding"
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class ChestXrayDataset(Dataset):

    def __init__(self, csv_file, image_folder):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_name = row["Image Index"]

        image_path = os.path.join(
            self.image_folder,
            image_name
        )

        image = Image.open(image_path).convert("RGB")

        image = transform(image)

        labels = row["Finding Labels"]

        target = []

        for label in LABELS:

            if label in labels:
                target.append(1.0)
            else:
                target.append(0.0)

        target = torch.tensor(
            target,
            dtype=torch.float32
        )

        return image, target


if __name__ == "__main__":

    dataset = ChestXrayDataset(
        "dataset/train.csv",
        "dataset/images"
    )

    image, target = dataset[0]

    print("Image shape:", image.shape)
    print("Labels:", target)
    print("Number of images:", len(dataset))