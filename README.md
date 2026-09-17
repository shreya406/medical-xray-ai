# Medical X-Ray AI

A deep learning-based web application for analyzing chest X-ray images and generating predictions for multiple conditions.

## About the Project

This project uses deep learning and transfer learning to analyze chest X-ray images. A ResNet18-based model is trained on a selected subset of the NIH ChestX-ray14 dataset.

The application allows a user to upload a chest X-ray image through a web interface and receive prediction scores for multiple conditions.

## Conditions

The model currently analyzes:

- Pneumonia
- Pneumothorax
- Cardiomegaly
- Effusion
- Atelectasis
- No Finding

## Technologies Used

- Python
- PyTorch
- Torchvision
- Flask
- Flask-CORS
- HTML
- CSS
- JavaScript
- Git & GitHub

## Model

The project uses a pretrained ResNet18 architecture with transfer learning.

The final classification layer was modified to predict six classes.

## Dataset

The project uses the NIH ChestX-ray14 dataset.

For this project, a balanced subset of the dataset was created and divided into training, validation, and testing sets.

The X-ray images are not included in this repository because of their size. They are excluded using `.gitignore`.

## Project Structure

```text
medical-xray-ai
│
├── backend
│   └── app.py
│
├── dataset
│   ├── Data_Entry_2017_v2020.csv
│   ├── balanced_data.csv
│   ├── selected_data.csv
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── frontend
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── learning
│   ├── check_dataset.py
│   ├── check_images.py
│   ├── dataset_loader.py
│   └── download_one.py
│
├── model
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── chest_xray_model.pth
│
└── .gitignore
