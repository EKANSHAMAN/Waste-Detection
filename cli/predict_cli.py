import argparse
import cv2
import numpy as np
import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model("mobilenetv2_waste_classifier.h5")
class_names = [
    'battery', 'biological', 'cardboard', 'clothes',
    'glass', 'green-glass', 'metal', 'paper',
    'plastic', 'shoes', 'trash', 'white-glass'
]
  # Replace if different
img_size = 224

def predict(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found -> {image_path}")
        return

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Unable to read the image. Ensure it's a valid image format.")
        return

    img = cv2.resize(img, (img_size, img_size))
    img_array = np.expand_dims(img / 255.0, axis=0)

    # Predict
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    label = class_names[pred_index]
    confidence = float(predictions[0][pred_index]) * 100

    print(f"\n🧠 Prediction: {label}")
    print(f"✅ Confidence: {confidence:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waste Classification CLI Tool")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()
    predict(args.image_path)
