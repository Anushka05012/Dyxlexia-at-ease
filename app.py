from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import io
from PIL import Image
import json
import os


app = Flask(__name__)
CORS(app)

PROGRESS_FILE = 'progress_data.json'

def read_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {}
    with open(PROGRESS_FILE, 'r') as f:
        return json.load(f)

def write_progress(data):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(data, f, indent=2)




# Load your EMNIST model
model = load_model("emnist_model.h5")
print("✅ Model loaded successfully!")

# EMNIST ByClass label map (adjust based on your model)
label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract base64 image from request
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])

        # Load image via PIL and convert to grayscale
        raw_image = Image.open(io.BytesIO(img_data)).convert('L')

        # Convert PIL image to OpenCV format
        img_array = np.array(raw_image)

        # Resize to 28x28 using INTER_AREA (better for downscaling)
        img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)

        # Invert: white background → black, black writing → white (EMNIST format)
        img_array = 255 - img_array

        # Apply blur to unify strokes, then threshold to remove artifacts
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        _, img_array = cv2.threshold(img_array, 80, 255, cv2.THRESH_BINARY)

        # Save for debug
        cv2.imwrite("flask_processed_debug.png", img_array)
        print("✅ Processed image saved. Shape:", img_array.shape)

        # Normalize and reshape for model
        img_array = img_array.astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)[0]
        print("🔎 Prediction array:", prediction)

        predicted_index = np.argmax(prediction)
        predicted_char = label_map[predicted_index]
        confidence = float(prediction[predicted_index]) * 100

        feedback = "✔ Great job!" if confidence >= 80 else "✘ Try again!"

        return jsonify({
            "prediction": predicted_char,
            "similarity": confidence,
            "feedback": feedback
        })

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500
    
@app.route('/get_progress', methods=['GET'])
def get_progress():
    child = request.args.get('child')
    
    # Read progress data from file
    progress_data = read_progress()
    
    if child in progress_data:
        child_data = progress_data[child]
        letters_progress = child_data.get('letters_progress', 0)
        digits_progress = child_data.get('digits_progress', 0)
        return jsonify({"letters_progress": letters_progress, "digits_progress": digits_progress})
    else:
        return jsonify({"error": "Child not found"}), 404



@app.route('/update_progress', methods=['POST'])
def update_progress():
    data = request.get_json()
    child = data.get('child')
    progress_type = data.get('type')  # 'letters' or 'digits'
    progress_value = data.get('progress')
    
    # Read current progress data
    progress_data = read_progress()
    
    # Check if child exists, and if not, initialize them
    if child not in progress_data:
        progress_data[child] = {
            "letters_progress": 0,
            "digits_progress": 0
        }
    
    # Update the progress data for the selected child
    if progress_type == 'letters':
        progress_data[child]['letters_progress'] = progress_value
    elif progress_type == 'digits':
        progress_data[child]['digits_progress'] = progress_value
    
    # Write the updated progress data back to the file
    write_progress(progress_data)
    
    return jsonify({"status": "success"})

@app.route('/get_confusables', methods=['GET'])
def get_confusables():
    try:
        with open('confusables.json') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_confusables', methods=['POST'])
def predict_confusables():
    try:
        images = request.json['images']  # A list of base64 images
        expected = request.json['expected']  # A list of two expected characters
        
        results = []

        for base64_img, expected_char in zip(images, expected):
            img_data = base64.b64decode(base64_img.split(',')[1])
            raw_image = Image.open(io.BytesIO(img_data)).convert('L')
            img_array = np.array(raw_image)
            img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
            img_array = 255 - img_array
            img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
            _, img_array = cv2.threshold(img_array, 80, 255, cv2.THRESH_BINARY)
            img_array = img_array.astype("float32") / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            prediction = model.predict(img_array)[0]
            predicted_index = np.argmax(prediction)
            predicted_char = label_map[predicted_index]
            confidence = float(prediction[predicted_index]) * 100

            correct = predicted_char == expected_char
            feedback = "✔ Correct" if correct else f"✘ Got {predicted_char} instead of {expected_char}"

            results.append({
                "expected": expected_char,
                "predicted": predicted_char,
                "confidence": confidence,
                "feedback": feedback
            })

        return jsonify({"results": results})

    except Exception as e:
        print("❌ Error during confusables prediction:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':  # <-- fix typo from your version ('_main_')
    app.run(debug=True)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        raw_image = Image.open(io.BytesIO(img_data)).convert('L')
        img_array = np.array(raw_image)
        img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
        img_array = 255 - img_array
        img_array = cv2.GaussianBlur(img_array, (3,3), 0)
        _, img_array = cv2.threshold(img_array, 80, 255, cv2.THRESH_BINARY)
        img_array = img_array.astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_char = label_map[predicted_index]
        confidence = float(prediction[predicted_index]) * 100
        feedback = "✔ Great job!" if confidence >= 80 else "✘ Try again!"
        return jsonify({
            "prediction": predicted_char,
            "similarity": confidence,
            "feedback": feedback
        })
    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
