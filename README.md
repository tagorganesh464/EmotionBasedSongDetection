# Emotion-Based Music Recommendation System 🎵🎭

This project aims to detect human emotions using facial images and recommend music based on the detected emotion. The system leverages a Convolutional Neural Network (CNN) for emotion classification and integrates with the Spotify API to provide personalized music recommendations.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Results](#results)
- [License](#license)

---

## Overview
The project detects facial emotions in grayscale images and classifies them into one of seven categories. Based on the detected emotion, relevant music tracks are recommended using the Spotify API.

---

## Dataset
- **Training Data Directory**: `data/train`
- **Validation Data Directory**: `data/test`
- The images are preprocessed using the `ImageDataGenerator` from Keras to normalize pixel values by rescaling them to the [0,1] range.

---

## Model Architecture
The model is a Convolutional Neural Network (CNN) built using Keras' Sequential API. The architecture includes:

- **Input**: Grayscale images of size 48x48 pixels.
- **Layers**:
  - **Convolutional Layers**: Three sets of Conv2D layers with ReLU activation, followed by MaxPooling2D for down-sampling.
  - **Dropout**: Applied after pooling layers to reduce overfitting.
  - **Fully Connected Layers**: Flatten layer followed by a Dense layer with 1024 neurons.
  - **Output Layer**: A Dense layer with 7 neurons (softmax activation) for multi-class emotion classification.

---

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV

You can install the dependencies using:

```bash
pip install -r requirements.txt

