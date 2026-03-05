import os
import base64
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="web")

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_obstruction_model2.h5")
model = tf.keras.models.load_model(MODEL_PATH)
class_labels = ["with_mask", "without_mask"]

face_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


def preprocess_face(frame, x, y, w, h):
    """Crop lower face, apply CLAHE, convert to grayscale 3-channel."""
    lower_y = y + int(h * 0.4)
    lower_h = h - int(h * 0.4)
    face_img = frame[lower_y:lower_y + lower_h, x:x + w]
    if face_img.size == 0:
        return None
    face_img = cv2.resize(face_img, (224, 224))

    # CLAHE — sharpens mask edges regardless of mask color
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    face_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Grayscale tiled to 3 channels — removes color bias
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = np.stack([gray, gray, gray], axis=-1)
    face_img = face_img / 255.0
    return np.expand_dims(face_img, axis=0)


def detect_faces(gray, frame_width):
    """Try frontal then profile (both sides) detection."""
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(80, 80))
    if len(faces) > 0:
        return faces

    # Left profile
    profile = profile_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(80, 80))
    if len(profile) > 0:
        return profile

    # Right profile (flip frame)
    flipped = cv2.flip(gray, 1)
    pf = profile_cascade.detectMultiScale(flipped, scaleFactor=1.05, minNeighbors=5, minSize=(80, 80))
    if len(pf) > 0:
        return np.array([[frame_width - x - w, y, w, h] for (x, y, w, h) in pf])

    return []


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("web", filename)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image from browser canvas
    try:
        img_bytes = base64.b64decode(data["image"].split(",")[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Image decode failed: {str(e)}"}), 400

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, frame.shape[1])

    if len(faces) == 0:
        return jsonify({"status": "no_face", "results": []})

    results = []
    for (x, y, w, h) in faces[:1]:  # process only the primary face
        face_input = preprocess_face(frame, x, y, w, h)
        if face_input is None:
            continue

        pred       = model.predict(face_input, verbose=0)[0]
        class_idx  = int(np.argmax(pred))
        confidence = round(float(pred[class_idx]) * 100, 1)
        label      = class_labels[class_idx]

        results.append({
            "label":      label,
            "confidence": confidence,
            "box":        [int(x), int(y), int(w), int(h)]
        })

    return jsonify({"status": "ok", "results": results})


if __name__ == "__main__":
    print("Starting Face Mask Detection server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
