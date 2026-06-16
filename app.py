import os
import json
import uuid

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import keras

from tensorflow.keras.applications.mobilenet import preprocess_input


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
MODEL_PATH = os.path.join("model", "best_fruitfour_mobilenetv1_fruit_condition.keras")
CLASS_INDEX_PATH = os.path.join("model", "class_indices.json")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

CONFIDENCE_THRESHOLD = 0.70

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def patch_layer_from_config(layer_class, remove_keys):
    original_from_config = layer_class.from_config

    @classmethod
    def patched_from_config(cls, config):
        config = dict(config)

        for key in remove_keys:
            config.pop(key, None)

        return original_from_config(config)

    layer_class.from_config = patched_from_config


patch_layer_from_config(
    keras.layers.BatchNormalization,
    ["renorm", "renorm_clipping", "renorm_momentum", "quantization_config"]
)

patch_layer_from_config(
    keras.layers.Dense,
    ["quantization_config"]
)

patch_layer_from_config(
    keras.layers.Conv2D,
    ["quantization_config"]
)

patch_layer_from_config(
    keras.layers.DepthwiseConv2D,
    ["quantization_config"]
)


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)


with open(CLASS_INDEX_PATH, "r") as file:
    class_indices = json.load(file)

index_to_class = {value: key for key, value in class_indices.items()}


CLASS_INFO = {
    "freshapples": {
        "fruit": "Apel",
        "condition": "Segar",
        "condition_class": "fresh"
    },
    "freshbanana": {
        "fruit": "Pisang",
        "condition": "Segar",
        "condition_class": "fresh"
    },
    "freshoranges": {
        "fruit": "Jeruk",
        "condition": "Segar",
        "condition_class": "fresh"
    },
    "rottenapples": {
        "fruit": "Apel",
        "condition": "Busuk",
        "condition_class": "rotten"
    },
    "rottenbanana": {
        "fruit": "Pisang",
        "condition": "Busuk",
        "condition_class": "rotten"
    },
    "rottenoranges": {
        "fruit": "Jeruk",
        "condition": "Busuk",
        "condition_class": "rotten"
    }
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence_value = float(predictions[predicted_index])
    predicted_class = index_to_class[predicted_index]

    confidence_percent = confidence_value * 100

    if confidence_value < CONFIDENCE_THRESHOLD:
        return {
            "detected": False,
            "fruit": "Tidak terdeteksi",
            "condition": "-",
            "condition_class": "unknown",
            "raw_class": predicted_class,
            "confidence": f"{confidence_percent:.2f}%",
        }

    if predicted_class not in CLASS_INFO:
        return {
            "detected": False,
            "fruit": "Tidak terdeteksi",
            "condition": "-",
            "condition_class": "unknown",
            "raw_class": predicted_class,
            "confidence": f"{confidence_percent:.2f}%",
        }

    info = CLASS_INFO[predicted_class]

    return {
        "detected": True,
        "fruit": info["fruit"],
        "condition": info["condition"],
        "condition_class": info["condition_class"],
        "raw_class": predicted_class,
        "confidence": f"{confidence_percent:.2f}%"
    }


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        result=None,
        image_path=None,
        error=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    result = None
    image_path = None
    error = None

    if "file" not in request.files:
        error = "Tidak ada file yang diunggah."
        return render_template(
            "index.html",
            result=None,
            image_path=None,
            error=error
        )

    file = request.files["file"]

    if file.filename == "":
        error = "Silakan pilih gambar terlebih dahulu."
        return render_template(
            "index.html",
            result=None,
            image_path=None,
            error=error
        )

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        extension = original_filename.rsplit(".", 1)[1].lower()
        new_filename = f"{uuid.uuid4().hex}.{extension}"

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], new_filename)
        file.save(save_path)

        result = predict_image(save_path)
        image_path = new_filename

    else:
        error = "Format file tidak valid. Gunakan JPG, JPEG, PNG, atau WEBP."

    return render_template(
        "index.html",
        result=result,
        image_path=image_path,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)