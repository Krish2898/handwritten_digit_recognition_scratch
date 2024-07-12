# app.py

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
model = np.load('digit_recognizer_model.npz')
W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

# Define necessary functions
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Streamlit interface
st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(784, 1) / 255.

    st.image(image.reshape(28, 28), caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = make_predictions(image, W1, b1, W2, b2)
    st.write(f"Prediction: {prediction[0]}")
