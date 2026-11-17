
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "genetic_disorder_model.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = model.feature_names_in_.tolist()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Genetic Disorder Risk Classification API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid patient data"}), 400

        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=feature_columns)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        results = {
            str(label): round(float(probability), 4)
            for label, probability in zip(model.classes_, probabilities)
        }

        return jsonify({
            "Predicted Genetic Disorder": str(prediction),
            "Prediction Probabilities": results,
            "Note": "Educational prototype only. Not a medical diagnosis."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
