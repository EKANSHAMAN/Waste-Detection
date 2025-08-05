
# ♻ Waste Classification Using Machine Learning

A Machine Learning project to classify garbage waste into categories like plastic, glass, cardboard, etc. using a trained MobileNetV2 model. Includes a FastAPI backend for prediction and a simple web frontend for image uploads.

## 🔍 Features

- Image classification using MobileNetV2 (Transfer Learning)
- Real-time webcam detection using OpenCV
- FastAPI-powered backend API
- CLI prediction script
- Free deployment on Render
- Frontend for uploading images and viewing predictions

## 📁 Folder Structure

```
waste-classifier/
├── app/
│ ├── main.py # FastAPI app
│ ├── mobilenetv2_waste_classifier.h5 # Trained model
│ 
│
├── frontend/
│ └── index.html # Simple frontend to upload and predict image
│----predict_cli.py # CLI-based prediction
├── render.yaml # Render deployment configuration
├── requirements.txt # Python dependencies
└── README.md
```

## 🚀 Setup & Installation

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### CLI Usage
```bash
cd cli
python predict_cli.py --image path/to/image.jpg
```

## 🌐 API Usage

**POST /predict**  
Upload an image and get prediction.

Request: `multipart/form-data` with key `file`  
Response:
```json
{
  "predicted_class": "plastic",
  "confidence": 95.32
}
```

## 🖼 Frontend

- Located in `/frontend/index.html`
- Uploads image and sends request to `/predict` endpoint.

Update the backend URL in JS:
```js
const BACKEND_URL = "http://localhost:8000/predict"; // or your Render URL
```

## ☁ Deployment on Render

1. Push project to GitHub
2. Log into [https://render.com](https://render.com)
3. Create new Web Service → Connect GitHub repo
4. Use `render.yaml` for settings
5. Wait for deployment

## 🧪 Requirements

See `requirements.txt`:
- fastapi
- uvicorn
- tensorflow
- numpy
- opencv-python

## 🤖 Model Training

Model trained on [Garbage Classification Dataset - Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

## 🙌 Contribution

Open to contributions and improvements!

## 📄 License

MIT License

## ✨ Acknowledgements

- TensorFlow for Transfer Learning with MobileNetV2
- FastAPI for backend API
- OpenCV for real-time detection
