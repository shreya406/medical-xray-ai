from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import ChestXrayModel

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [
    "Pneumonia",
    "Pneumothorax",
    "Cardiomegaly",
    "Effusion",
    "Atelectasis",
    "No Finding"
]

model = ChestXrayModel()
model.load_state_dict(
    torch.load(
        "model/chest_xray_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.route("/")
def home():
    return jsonify({
        "message": "Chest X-Ray AI API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["file"]

    image = Image.open(file).convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.sigmoid(output)[0]

    results = {}

    for label, probability in zip(LABELS, probabilities):
        results[label] = round(probability.item() * 100, 2)

    return jsonify(results)


if __name__ == "__main__":
    print("Starting Chest X-Ray AI Backend...")
    print("Using device:", device)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )