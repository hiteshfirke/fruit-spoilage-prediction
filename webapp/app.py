# app.py
from flask import Flask, render_template, request, jsonify
import os
import uuid

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------------------------
# PATHS & FLASK SETUP
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../webapp
PROJECT_ROOT = os.path.dirname(BASE_DIR)                     # .../fruit_spoilage_project3
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = os.path.join(DATASET_DIR, "banana_stage_mobilenet_final.keras")

app = Flask(__name__, static_folder="static", template_folder="templates")

# -------------------------------------------------
# LOAD MODEL ONCE
# -------------------------------------------------

print("[INFO] Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded.")

CLASS_NAMES = [
    "stage0_unripe",
    "stage1_ripe",
    "stage2_overripe",
    "stage3_rotten",
]


# -------------------------------------------------
# HELPER FUNCTIONS  (SHARED WITH CLI LOGIC)
# -------------------------------------------------

def preprocess_for_mobilenet(image_path: str) -> np.ndarray:
    """
    Load an image, resize to 224x224, apply MobileNetV2 preprocessing.
    Returns a (1, 224, 224, 3) float32 batch.
    """
    img = load_img(image_path, target_size=(224, 224))
    arr = img_to_array(img)
    # keep a copy BEFORE preprocess for brightness computation
    arr_raw = arr.astype("float32") / 255.0
    brightness = float(arr_raw.mean())

    arr = preprocess_input(arr)  # MobileNetV2 preprocessing
    arr = np.expand_dims(arr, axis=0)
    return arr, brightness


def estimate_days(stage_label: str, temp_c: float, hum: float) -> float:
    """
    Return a single continuous 'days_left' value, instead of a range.
    Logic: start from a base range per stage, take the midpoint, then
    adjust with temperature & humidity factors (same idea as before).
    """

    # Base ranges per stage at normal room conditions
    base_ranges = {
        "stage0_unripe": (4.0, 7.0),
        "stage1_ripe": (1.0, 3.0),
        "stage2_overripe": (0.0, 2.0),
        "stage3_rotten": (0.0, 0.0),
    }

    dmin, dmax = base_ranges.get(stage_label, (0.0, 0.0))
    # Mid-point is our “ideal” days left before environmental adjustment
    base_days = (dmin + dmax) / 2.0

    # Temperature factor – hotter → faster spoilage
    if temp_c > 30:
        factor = 0.5
    elif temp_c > 28:
        factor = 0.65
    elif temp_c > 26:
        factor = 0.8
    elif temp_c < 18:
        factor = 1.3
    else:
        factor = 1.0

    # Humidity small adjustment
    if hum > 80:
        factor *= 0.85
    elif hum < 40:
        factor *= 0.9

    days_left = max(0.0, base_days * factor)
    return round(days_left, 1)  # e.g. 2.8



def run_model(image_path: str, temp_c: float, hum: float):
    """
    Full pipeline: preprocessing, inference, brightness override,
    and shelf-life estimation. Returns a result dict.
    """
    # Preprocess and get brightness similar to CLI
    batch, brightness = preprocess_for_mobilenet(image_path)

    preds = model.predict(batch)[0]  # shape (4,)
    idx = int(np.argmax(preds))
    stage_label = CLASS_NAMES[idx]
    confidence = float(preds[idx])

    # Brightness-based override: avoid false "rotten" on bright backgrounds.
    # Threshold is approximate but matches CLI behaviour.
    BRIGHT_THRESH = 0.40
    if stage_label == "stage3_rotten" and brightness > BRIGHT_THRESH:
        stage_label = "stage2_overripe"

    days_left = estimate_days(stage_label, temp_c, hum)

    return {
        "stage_label": stage_label,
        "confidence": confidence,
        "days_left": days_left,
        "temperature": temp_c,
        "humidity": hum,
        "brightness": brightness,
    }




# -------------------------------------------------
# ROUTES
# -------------------------------------------------

@app.route("/")
def starter_page():
    # Starter / landing page (fruit market video background)
    return render_template("index.html")


@app.route("/main")
def main_page():
    # Main predictor UI
    return render_template("main.html")


@app.route("/architecture")
def architecture_page():
    # Model & research architecture page
    return render_template("architecture.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("image")
        temp_c = float(request.form.get("temperature", 24))
        hum = float(request.form.get("humidity", 60))

        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        # --- Save upload ---
        ext = os.path.splitext(file.filename)[1].lower()
        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        # --- Run model ---
        raw = run_model(save_path, temp_c, hum)
        print("[DEBUG] raw_run_model result:", raw)

        # Pull values out of raw, with safe defaults
        stage_label = raw.get("stage_label", "unknown_stage")
        confidence = float(raw.get("confidence", 0.0))      # 0–1
        days_left = float(raw.get("days_left", 0.0))        # single exact value
        temp_out = float(raw.get("temperature", temp_c))
        hum_out = float(raw.get("humidity", hum))

        # Build clean JSON for the frontend
        result = {
            "stage_label": stage_label,
            "confidence": confidence,   # 0–1
            "days_exact": days_left,    # single exact value
            "days_min": days_left,      # keep same if you don’t have range
            "days_max": days_left,      # keep same if you don’t have range
            "temperature": temp_out,
            "humidity": hum_out,
        }

        print("[DEBUG] sending JSON to client:", result)
        return jsonify(result)

    except Exception as e:
        print("[ERROR] /predict failed:", repr(e))
        return jsonify({"error": "Server error"}), 500

# -------------------------------------------------
# MAIN
# -------------------------------------------------

if __name__ == "__main__":
    # Use the same port you’ve been using in the browser (5500 in your screenshots).
    app.run(host="127.0.0.1", port=5500, debug=True)
