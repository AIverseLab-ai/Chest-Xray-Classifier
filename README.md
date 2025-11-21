## Chest X-Ray Pneumonia Classification Using CNN (TensorFlow)

This project uses a Convolutional Neural Network (CNN) to classify Chest X-Ray images as Normal or Pneumonia.
The model was trained in Google Colab using TensorFlow 2.19.0 and TPU/GPU acceleration.

# Project Highlights

Built end-to-end deep learning pipeline in Colab

Preprocessed ~5,800 chest X-ray images

Used ImageDataGenerator for augmentation

CNN model with Conv2D + MaxPooling layers

Achieved ~89% accuracy in 5 epochs (can be improved further)

TPU/GPU supported training

Clean, step-by-step notebook

Beginner-friendly, internship-ready project

## Dataset

The dataset used is Chest X-Ray Pneumonia Dataset from Kaggle:

## Dataset Link:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Important:
The dataset is large (~1GB), so it is not uploaded in the repository.
To train the model, download it from Kaggle or load it using Kaggle API.

## Model Architecture

The CNN model includes:

Conv2D (32 filters)

Conv2D (64 filters)

Conv2D (128 filters)

MaxPooling layers

Flatten

Dense layers

Sigmoid output for binary classification

## Training Details

Image size: 150 × 150

Batch size: 32

Optimizer: Adam

Loss: binary_crossentropy

Epochs Trained: 5

Accuracy Achieved: ~89%

You can increase epochs or add more layers for better accuracy.

## Requirements

The project was developed using:

Python 3.x  
TensorFlow 2.19.0  
Google Colab  
TPU/GPU Runtime (recommended)


No extra installations needed in Colab.

## How to Run the Notebook

Open the notebook in Google Colab

Upload the dataset ZIP file OR connect Kaggle API

Extract dataset

Run all cells step-by-step

Model will begin training and show accuracy/loss

## Results

Training accuracy: ~89%

Validation accuracy: ~85–88%

Model detects pneumonia patterns from X-Rays

You can improve results by:

Using more epochs

Adding Dropout layers

Using Transfer Learning (like MobileNetV2, VGG16, etc.)

## Future Improvements

Implement Transfer Learning

Add Grad-CAM visualization

Deploy with Streamlit / Flask

Add real-time X-ray prediction interface
