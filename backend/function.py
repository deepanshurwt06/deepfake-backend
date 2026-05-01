import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50

detector=MTCNN()

resnet = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

def extract_faces_from_video(video_path, max_frames=15):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open video")
        return []

    faces = []
    frame_index = 0

    while cap.isOpened() and len(faces) < max_frames:

        ret, frame = cap.read()
        if not ret:
            break

        # skip frames for speed
        if frame_index % 2 != 0:
            frame_index += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = detector.detect_faces(rgb_frame)

        if len(detections) > 0:

            # take most confident face
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

            x, y, w, h = detections[0]['box']

            # fix negative coordinates
            x, y = max(0, x), max(0, y)

            face = frame[y:y+h, x:x+w]

            try:
                face = cv2.resize(face, (224, 224))
                faces.append(face)
            except:
                pass

        frame_index += 1

    cap.release()

    # 🔥 pad if less than required
    if len(faces) < max_frames and len(faces) > 0:
        last_face = faces[-1]
        while len(faces) < max_frames:
            faces.append(last_face)

    return faces

def extract_features(faces):

    features = []

    for face in faces:
        img = np.expand_dims(face, axis=0)
        img = preprocess_input(img)

        feature = resnet.predict(img, verbose=0)
        features.append(feature[0])

    return np.array(features)