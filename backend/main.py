from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import cv2
import io

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model("mobilenetv2_waste_classifier.h5")
class_names = [
    'battery', 'biological', 'cardboard', 'clothes',
    'glass', 'green-glass', 'metal', 'paper',
    'plastic', 'shoes', 'trash', 'white-glass'
]
  # Replace if different
img_size = 224

@app.get("/")
def read_root():
    return {"message": "Waste Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    # Convert image to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    img = cv2.resize(img, (img_size, img_size))
    img_array = np.expand_dims(img / 255.0, axis=0)

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    label = class_names[pred_index]
    confidence = float(predictions[0][pred_index])

    return {
        "predicted_class": label,
        "confidence": round(confidence * 100, 2)
    }
