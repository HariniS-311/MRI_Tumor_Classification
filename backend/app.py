
import os
import uuid
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from preprocessing import preprocess_image, extract_features

app = Flask(__name__)
CORS(app)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "svm_hog_only.pkl"
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained HOG-only SVM model
model = joblib.load(MODEL_PATH)

CLASSES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "NeuroScan AI Backend is running!"
    })


@app.route("/api/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported image format"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        file.save(image_path)

        # Apply the same preprocessing used during training
        image = preprocess_image(image_path)

        # Extract HOG + GLCM features
        features = extract_features(image)

        # HOG-only model uses the first 8100 features
        hog_features = features[:8100].reshape(1, -1)

        prediction = model.predict(hog_features)[0]
        predicted_class = CLASSES[int(prediction)]

        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "message": "MRI prediction completed successfully"
        })

    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
