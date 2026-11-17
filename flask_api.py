
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("svm_model.pkl")

@app.route("/")
def home():
    return "Genetic Disorder Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_data = pd.DataFrame(data)

    prediction = model.predict(input_data)

    return jsonify({
        "Predicted Disorder Subclass": prediction.tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
