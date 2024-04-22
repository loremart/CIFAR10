import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
#verion 1.2

# Load the trained CNN model for CIFAR-10
model = models.load_model('cifar10_model.h5')
model.compile_metrics = model.metrics

# Define the classes for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255.0

    # Perform prediction using the model
    predictions = model.predict(image)
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]
    return class_name


def load_and_predict_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Perform prediction on the selected image
        prediction = predict_image(file_path)
        # Display the image in the window
        image = Image.open(file_path)
        image = image.resize((200, 200))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        # Show the prediction result
        result_label.config(text=f"Predicted class: {prediction}")


# Create the main window
root = tk.Tk()
root.title("Image Classifier")

# Button to load and predict the image
load_button = tk.Button(root, text="Load and Predict", command=load_and_predict_image)
load_button.pack()

# Label to display the image
image_label = tk.Label(root)
image_label.pack()

# Label to display the prediction result
result_label = tk.Label(root, text="")
result_label.pack()

# Run the window loop
root.mainloop()
