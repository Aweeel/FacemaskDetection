import cv2
import tensorflow as tf
import numpy as np
from collections import deque

model = tf.keras.models.load_model("face_obstruction_model2.h5")

class_labels = ["mask_worn_incorrectly", "with_mask", "without_mask"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

prediction_buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        # --- Face crop ---
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # --- Lighting normalization ---
        lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
        face_img = cv2.filter2D(face_img, -1, kernel)


        # --- Normalize ---
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        # --- Predict ---
        pred = model.predict(face_img, verbose=0)[0]
        prediction_buffer.append(pred)

        avg_pred = np.mean(prediction_buffer, axis=0)
        class_idx = np.argmax(avg_pred)
        confidence = avg_pred[class_idx]
        label = class_labels[class_idx]

        # --- Confidence threshold ---
        if label == "with_mask" and confidence < 0.45:
            pass
        elif confidence < 0.6:
            label = "uncertain"

        # --- Draw ---
        color = (0, 255, 0)
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
