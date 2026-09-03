# ♻️ Waste Classification CNN
## 🌐 Live Demo

[🚗 Car Price Prediction App](https://car-price-ml.streamlit.app/)
A deep learning image classification project that classifies waste images using a Convolutional Neural Network built with PyTorch.

## 🚀 Project Overview

This project uses a CNN to classify waste images into multiple categories.

The pipeline includes:

- Image preprocessing
- Data augmentation
- Custom PyTorch Dataset
- DataLoader
- CNN architecture
- Model training
- Validation
- Test evaluation
- Confusion matrix
- Classification report
- Single-image prediction
- Streamlit deployment

## 🧠 Technologies

- Python
- PyTorch
- Torchvision
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow
- Streamlit

## 🏗️ CNN Architecture

Input Image
→ Conv2D
→ ReLU
→ MaxPool
→ Conv2D
→ ReLU
→ MaxPool
→ Flatten
→ Linear
→ ReLU
→ Linear
→ Output Classes

## 📊 Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Per-class accuracy

## 🌐 Streamlit App

The application allows users to upload an image and receive:

- Predicted waste class
- Confidence score
- Top 3 predictions

## 📁 Project Structure

```text
cnn-waste-classification/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   └── best_waste_cnn.pth
│
├── artifacts/
│   └── class_names.json
│
└── notebooks/
    └── cnn_waste_classification.ipynb
