# digit_similarity.py
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# Load trained model
model_path = "mnist_digit_model.keras"
model = None
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Warning: Model not found! Digit prediction will not work.")

def recognize_digit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    img = cv2.resize(img, (28, 28)).astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    if model:
        predictions = model.predict(img, verbose=0)
        return np.argmax(predictions)
    else:
        print("Model not loaded.")
        return None

def load_standard_images(folder_path):
    standard_images = {}
    if not os.path.exists(folder_path):
        print(f"Error: Standard image folder '{folder_path}' not found!")
        return standard_images

    for filename in os.listdir(folder_path):
        digit = filename.split(".")[0]
        if not digit.isdigit():
            continue

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (28, 28))
        standard_images[int(digit)] = img
    return standard_images

def compare_digit(user_image_path, standard_folder):
    standard_images = load_standard_images(standard_folder)
    if not standard_images:
        return None, None

    user_img = cv2.imread(user_image_path, cv2.IMREAD_GRAYSCALE)
    if user_img is None:
        print(f"Error: Unable to load user image at {user_image_path}")
        return None, None

    user_img = cv2.resize(user_img, (28, 28)).flatten().reshape(1, -1)

    predicted_digit = recognize_digit(user_image_path)
    if predicted_digit is None:
        return None, None

    standard_img = standard_images.get(predicted_digit)
    if standard_img is None:
        print(f"No standard image found for digit {predicted_digit}")
        return None, None

    standard_img = standard_img.flatten().reshape(1, -1)
    similarity = cosine_similarity(user_img, standard_img)[0][0]

    return predicted_digit, round(similarity * 100, 2)
