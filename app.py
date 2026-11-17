
import streamlit as st
import pandas as pd
import joblib
import requests

st.set_page_config(
    page_title="Genetic Disorder Risk Classification",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Disorder Risk Classification System")
st.write("Enter patient details to predict the genetic disorder category.")

# Load trained model to identify input features
@st.cache_resource
def load_model():
    return joblib.load("genetic_disorder_model.pkl")

model = load_model()

# Get numeric and categorical feature names
preprocessor = model.named_steps["preprocessor"]

numeric_features = []
for name, transformer, columns in preprocessor.transformers_:
    if name == "num":
        numeric_features = list(columns)

feature_columns = list(model.feature_names_in_)

# Flask API URL
st.sidebar.header("Flask API Settings")
api_url = st.sidebar.text_input(
    "API URL",
    value="http://127.0.0.1:5000"
)

st.info("Fill in the details you know. Unknown fields can be left blank.")

patient_data = {}

with st.form("patient_form"):
    st.subheader("Patient Details")

    col1, col2 = st.columns(2)

    for index, feature in enumerate(feature_columns):
        column = col1 if index % 2 == 0 else col2

        with column:
            value = st.text_input(
                feature,
                key=f"field_{index}",
                placeholder="Leave blank if unknown"
            )

            if value.strip():
                if feature in numeric_features:
                    try:
                        patient_data[feature] = float(value)
                    except ValueError:
                        st.warning(f"Enter a valid number for {feature}.")
                else:
                    patient_data[feature] = value.strip()

    submitted = st.form_submit_button("Predict Genetic Disorder")

if submitted:
    if not patient_data:
        st.warning("Please enter at least one patient detail.")
    else:
        try:
            with st.spinner("Getting prediction..."):
                response = requests.post(
                    f"{api_url.rstrip('/')}/predict",
                    json=patient_data,
                    timeout=30
                )

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction completed!")

                st.subheader("Prediction Result")
                st.write(
                    "**Predicted Genetic Disorder:**",
                    result["Predicted Genetic Disorder"]
                )

                st.subheader("Prediction Probabilities")
                probabilities = result["Prediction Probabilities"]

                for disorder, probability in probabilities.items():
                    st.write(f"**{disorder}**")
                    st.progress(float(probability))
                    st.write(f"{probability * 100:.2f}%")

                st.caption(result.get("Note", ""))

            else:
                st.error(f"API Error: {response.text}")

        except requests.exceptions.RequestException:
            st.error(
                "Unable to connect to Flask API. "
                "Check whether the Flask server is running."
            )

st.divider()
st.caption(
    "Educational prototype only. This system is not a medical diagnosis."
)
