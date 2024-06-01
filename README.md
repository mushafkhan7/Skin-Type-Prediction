# Skin Type Prediction

Skin Type Prediction is a Python application that predicts the skin type (dry, normal, or oily) based on uploaded images using convolutional neural networks (CNNs). This project leverages the TensorFlow and scikit-learn libraries for model building and evaluation.

## Features

- **Data Loading and Preprocessing:** The application loads and preprocesses skin images from directories using OpenCV.
- **Convolutional Neural Network Model:** A CNN model is constructed using TensorFlow's Keras API for image classification.
- **Data Augmentation:** Training data is augmented using TensorFlow's ImageDataGenerator to improve model robustness.
- **Graphical User Interface (GUI):** Skin images can be uploaded via a simple GUI built with tkinter for prediction.

## Requirements

- Python 3.x
- TensorFlow
- scikit-learn
- matplotlib
- tkinter
- OpenCV
- Pillow

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/mushafkhan7/skin-type-prediction.git
