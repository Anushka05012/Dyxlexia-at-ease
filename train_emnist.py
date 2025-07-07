
import numpy as np
import matplotlib.pyplot as plt
from emnist import extract_training_samples, extract_test_samples
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load EMNIST "byclass" (digits + letters)
(train_ds, test_ds), ds_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,  # Returns (image, label) tuples
    with_info=True,      # Provides dataset metadata
)

# Convert to NumPy arrays (optional, but useful for Keras)
def dataset_to_numpy(ds):
    images, labels = [], []
    for img, lbl in tfds.as_numpy(ds):
        images.append(img)
        labels.append(lbl)
    return np.array(images), np.array(labels)

X_train, y_train = dataset_to_numpy(train_ds)
X_test, y_test = dataset_to_numpy(test_ds)

# EMNIST images are transposed (rotated 90°), so we fix them
X_train = np.transpose(X_train, (0, 2, 1, 3))
X_test = np.transpose(X_test, (0, 2, 1, 3))

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Check shapes
print("Training data shape:", X_train.shape)  # Should be (N, 28, 28, 1)
print("Test data shape:", X_test.shape)      # Should be (N, 28, 28, 1)
print("Number of classes:", ds_info.features['label'].num_classes)  # 62 (digits + uppercase + lowercase)

import matplotlib.pyplot as plt

# Define class names (EMNIST 'byclass' has 62 classes: 0-9, A-Z, a-z)
class_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential([
    # Convolutional Block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Convolutional Block 2
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Fully Connected Layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(62, activation='softmax')  # 62 classes for EMNIST 'byclass'
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if labels are one-hot encoded
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=50,  # Will stop early if validation loss doesn't improve
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Evaluate on training set (use a subset for faster evaluation if dataset is large)
train_loss, train_acc = model.evaluate(X_train[:10000], y_train[:10000], verbose=0)

# Evaluate on full test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n=== Final Model Performance ===")
print(f"Training Accuracy:   {train_acc:.2%}")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")  # Last epoch's val accuracy
print(f"Test Accuracy:       {test_acc:.2%}")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def test_custom_image(model, class_names):
    # Upload image
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded!")
        return
    
    img_path = next(iter(uploaded))
    
    # Preprocess image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image!")
        return
    
    # Display original
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.title("Original Uploaded Image")
    plt.axis('off')
    plt.show()
    
    # Preprocessing pipeline (must match training)
    img = cv2.resize(img, (28, 28))
    img = 255 - img  # Invert colors (EMNIST uses white-on-black)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Add batch and channel dims
    
    # Predict
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # Display only the top prediction
    print("\n=== Prediction Result ===")
    print(f"Predicted Character: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2%}")
    
    # Visualize preprocessed image
    plt.figure(figsize=(3,3))
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Predicted: {class_names[predicted_class]}")
    plt.axis('off')
    plt.show()

# Usage:
test_custom_image(model,class_names)

model.save("emnist_model.h5")