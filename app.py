from fastapi import FastAPI, UploadFile, File
import shutil
import os
import numpy as np
from tensorflow.keras.models import load_model
from function import extract_faces_from_video, extract_features
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
app = FastAPI()

# load model once
model = load_model("model_v2.h5", compile=False)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Deepfake Detection API Running"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    # save uploaded video
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        faces = extract_faces_from_video(file_path)

        if len(faces) == 0:
            return {"result": "No faces detected"}

        features = extract_features(faces)

        if features.shape[0] == 0:
            return {"result": "Feature extraction failed"}

        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)[0][0]

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
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

