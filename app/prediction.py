"""
Prediction Module for MNIST Digit Recognition
==============================================
Image processing and model prediction utilities.
"""

from PIL import Image
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional
import streamlit as st


@st.cache_resource
def load_model():
    return joblib.load("model/mnist_model.pkl")


def process_canvas_image(canvas_data: np.ndarray) -> np.ndarray:
    """
    Process canvas image data to MNIST-compatible 28x28 format.
    
    Args:
        canvas_data: Raw canvas image data (RGBA)
    
    Returns:
        Processed 28x28 grayscale image
    """
    # Convert to grayscale
    canvas_array = canvas_data[:, :, :3].astype("uint8")
    gray = np.dot(canvas_array[..., :3], [0.299, 0.587, 0.114]).astype("uint8")
    
    # Threshold to binary
    gray[gray < 50] = 0
    gray[gray >= 50] = 255
    
    # Find bounding box of digit
    coords = np.column_stack(np.where(gray > 0))
    if coords.size != 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        digit = gray[y_min:y_max+1, x_min:x_max+1]
    else:
        digit = gray
    
    # Resize maintaining aspect ratio to fit in 20x20 box
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(w * (20 / h)))
    else:
        new_w = 20
        new_h = max(1, int(h * (20 / w)))
    
    digit_resized = Image.fromarray(digit).resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )
    digit_resized = np.array(digit_resized)
    
    # Center in 28x28 image
    final_img = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
    
    return final_img


def predict_digit(model, image: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """
    Predict digit from processed image.
    
    Args:
        model: Trained model
        image: 28x28 grayscale image
    
    Returns:
        Tuple of (predicted_digit, confidence, all_probabilities)
    """
    # Normalize and add batch dimension
    image_array = image.astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict
    prediction = model.predict(image_array)
    predicted_class = int(np.argmax(prediction))
    confidence = float(prediction[0][predicted_class]) * 100
    all_probs = prediction[0] * 100
    
    return predicted_class, confidence, all_probs
