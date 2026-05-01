from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from function import extract_faces_from_video, extract_features

# -----------------------------
# Reduce TensorFlow logs
# -----------------------------
tf.get_logger().setLevel('ERROR')

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI()

# -----------------------------
# Enable CORS (for frontend)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Setup directories
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load Model (only once)
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, "model_v2.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file not found!")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Deepfake Detection API Running"}

# -----------------------------
# Prediction API
# -----------------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    # generate unique filename
    unique_name = str(uuid.uuid4()) + "_" + file.filename
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    # save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Step 1: Extract faces
        faces = extract_faces_from_video(file_path)

        if len(faces) == 0:
            return {"result": "No faces detected", "confidence": 0}

        # Step 2: Extract features
        features = extract_features(faces)

        if features.shape[0] == 0:
            return {"result": "Feature extraction failed", "confidence": 0}

        # Step 3: Prepare for model
        features = np.expand_dims(features, axis=0)

        # Step 4: Predict
        prediction = model.predict(features)[0][0]

        # Step 5: Decision
        if prediction > 0.6:
            result = "Fake Video"
        else:
            result = "Real Video"

        return {
            "result": result,
            "confidence": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
