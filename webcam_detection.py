#  C:\Users\PC\AppData\Local\Python\pythoncore-3.11-64\python.exe webcam_detection.py

import cv2
import tensorflow as tf
import numpy as np
from collections import deque

model = tf.keras.models.load_model("models/face_obstruction_model2.h5")

class_labels = ["with_mask", "without_mask"]

# alt2 cascade is more tolerant of partial occlusion (e.g. masked lower face)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)
# profile cascade catches left/right side views
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

cap = cv2.VideoCapture(0)

prediction_buffer = deque(maxlen=20)  # more frames = smoother average
box_buffer = deque(maxlen=6)  # smooth bounding box over last 6 frames
smoothed_confidence = 0.0  # exponentially smoothed display value
smoothed_label = None
SMOOTH_ALPHA = 0.1  # lower = slower/smoother, higher = more responsive

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(80, 80)
    )

    # Also detect profile faces (side views)
    if len(faces) == 0:
        profile_faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=5, minSize=(80, 80)
        )
        if len(profile_faces) == 0:
            # Try flipped frame for the other side
            profile_faces_flipped = profile_cascade.detectMultiScale(
                cv2.flip(gray, 1), scaleFactor=1.05, minNeighbors=5, minSize=(80, 80)
            )
            if len(profile_faces_flipped) > 0:
                fw = frame.shape[1]
                profile_faces = np.array([
                    [fw - x - w, y, w, h] for (x, y, w, h) in profile_faces_flipped
                ])
        if len(profile_faces) > 0:
            faces = profile_faces

    # Clear buffer when no face detected to avoid stale predictions
    if len(faces) == 0:
        prediction_buffer.clear()
        box_buffer.clear()

    for (x, y, w, h) in faces[:1]:  # only process the largest/first face
        # --- Smooth bounding box by averaging over recent frames ---
        box_buffer.append((x, y, w, h))
        x, y, w, h = np.mean(box_buffer, axis=0).astype(int)
        # --- Use lower 60% of face (nose-to-chin) where mask is worn ---
        # This prevents skin color in forehead/eyes from polluting the prediction
        lower_y = y + int(h * 0.4)
        lower_h = h - int(h * 0.4)
        face_img = frame[lower_y:lower_y + lower_h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))

        # --- CLAHE to enhance mask edges/texture regardless of mask color ---
        face_lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(face_lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        face_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # --- Convert to grayscale tiled to 3 channels ---
        # This removes ALL color information, forcing the model to use only
        # texture, edges, and shape — making skin-colored masks detectable
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = np.stack([gray_face, gray_face, gray_face], axis=-1)
        # --- Normalize ---
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        # --- Predict ---
        pred = model.predict(face_img, verbose=0)[0]
        prediction_buffer.append(pred)

        avg_pred = np.mean(prediction_buffer, axis=0)
        class_idx = np.argmax(avg_pred)
        raw_confidence = avg_pred[class_idx]
        label = class_labels[class_idx]

        # Exponential moving average on confidence — eases the number on screen
        smoothed_confidence = SMOOTH_ALPHA * raw_confidence + (1 - SMOOTH_ALPHA) * smoothed_confidence
        confidence = smoothed_confidence

        # --- Confidence threshold ---
        if confidence < 0.55:
            label = "uncertain"

        # --- Draw ---
        if label == "with_mask":
            color = (0, 255, 0)   # green
        elif label == "without_mask":
            color = (0, 0, 255)   # red
        else:
            color = (0, 165, 255) # orange for uncertain
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text = f"{label}: {confidence*100:.1f}%"
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Face Obstruction Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
