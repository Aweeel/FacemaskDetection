import base64
import os
import sys
import threading
import webbrowser
import shutil
import sqlite3
from datetime import datetime

import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if hasattr(sys, "_MEIPASS"):
    BASE_DIR = sys._MEIPASS


APP_DIR = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts):
    return os.path.join(BASE_DIR, *parts)


WEB_DIR = resource_path("web")
MODEL_PATH = resource_path("models", "face_obstruction_model2.h5")
BUILTIN_DB_PATH = resource_path("detections.db")
DB_PATH = os.path.join(APP_DIR, "detections.db")


def ensure_db_file():
    """Copy the bundled database next to the executable on first launch."""
    if os.path.exists(DB_PATH):
        return

    if os.path.exists(BUILTIN_DB_PATH):
        shutil.copy2(BUILTIN_DB_PATH, DB_PATH)

app = Flask(__name__, static_folder=WEB_DIR)

# ── Database Setup ────────────────────────────────────────────
def init_db():
    """Create detections table if it doesn't exist."""
    ensure_db_file()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            camera INTEGER NOT NULL,
            face_index INTEGER NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            x INTEGER,
            y INTEGER,
            w INTEGER,
            h INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_detection(camera, face_index, label, confidence, x, y, w, h):
    """Log a single face detection to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO detections (timestamp, camera, face_index, label, confidence, x, y, w, h)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, camera, face_index, label, confidence, x, y, w, h))
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Load model once at startup
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
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(WEB_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    camera = data.get("camera", 0)  # camera index (0-3)

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
    for face_idx, (x, y, w, h) in enumerate(faces[:3]):  # process up to 3 faces
        face_input = preprocess_face(frame, x, y, w, h)
        if face_input is None:
            continue

        pred       = model.predict(face_input, verbose=0)[0]
        class_idx  = int(np.argmax(pred))
        confidence = round(float(pred[class_idx]) * 100, 1)
        label      = class_labels[class_idx]

        # Log to database
        log_detection(camera, face_idx, label, confidence, int(x), int(y), int(w), int(h))

        results.append({
            "label":      label,
            "confidence": confidence,
            "box":        [int(x), int(y), int(w), int(h)]
        })

    return jsonify({"status": "ok", "results": results})


@app.route("/logs", methods=["GET"])
def get_logs():
    """Retrieve recent detections from database."""
    limit = request.args.get("limit", 100, type=int)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    logs = [dict(row) for row in rows]
    return jsonify({"logs": logs})


@app.route("/logs/clear", methods=["POST"])
def clear_logs():
    """Clear all logs from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    conn.commit()
    conn.close()
    return jsonify({"status": "cleared"})


def open_browser(url):
    webbrowser.open_new(url)


def main():
    url = "http://127.0.0.1:5000"
    threading.Timer(1.0, open_browser, args=(url,)).start()
    print(f"Starting Face Mask Detection server at {url}")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
