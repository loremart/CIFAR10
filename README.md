Image Classification using CIFAR-10 Dataset and GUI

This repository contains two Python scripts for image classification using the CIFAR-10 dataset and a Graphical User Interface (GUI) built with Tkinter:

CIFAR-10 Model Training Script: This script loads the CIFAR-10 dataset, preprocesses the images, builds, compiles, trains, and evaluates a Convolutional Neural Network (CNN) model for image classification. It uses TensorFlow and Keras for model development.
Image Classification GUI Script: This script allows users to load an image, preprocess it, and perform predictions using the trained CIFAR-10 CNN model. The GUI is created using Tkinter, a Python library for building user interfaces.

Files:
- model.py: Python script for training the CNN model on the CIFAR-10 dataset.
- main.py: Python script for building a GUI to perform image classification using the trained model.
- cifar10_model.h5: Pre-trained CNN model saved in Hierarchical Data Format (H5) after training on the CIFAR-10 dataset.

Usage:
CIFAR-10 Model Training Script
Run model.py to train the CNN model on the CIFAR-10 dataset.
After training, the script will display the training and validation accuracy plots, and print the test accuracy of the model.
Image Classification GUI Script
Ensure that the pre-trained model file cifar10_model.h5 is present in the same directory as main.py.
Run main.py to launch the GUI application.
Click on the "Load and Predict" button to load an image and see the predicted class displayed below the image.

Requirements:
- Python 3.x
- TensorFlow
- Keras
- Tkinter
- Matplotlib
- NumPy
- Pillow
