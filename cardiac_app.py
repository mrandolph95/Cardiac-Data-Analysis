import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify

cardiac_model = tf.keras.models.load_model("heart_model.h5")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_data = np.array(data["input"]).reshape(-1,-1)
        prediction = cardiac_model.predict(input_data).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
