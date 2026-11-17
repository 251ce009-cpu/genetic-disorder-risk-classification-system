
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Genetic Disorder Risk Classification",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("genetic_disorder_model.pkl")

model = load_model()

preprocessor = model.named_steps["preprocessing"]
numeric_features = []

for name, transformer, columns in preprocessor.transformers_:
    if name == "num":
        numeric_features = list(columns)

feature_columns = list(model.feature_names_in_)

st.title("🧬 Genetic Disorder Risk Classification")
st.write("Enter patient details to predict the genetic disorder category.")

patient_data = {}

with st.form("patient_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_columns):
        with (col1 if i % 2 == 0 else col2):
            value = st.text_input(
                feature,
                placeholder="Leave blank if unknown"
            )

        if value.strip():
            if feature in numeric_features:
                try:
                    patient_data[feature] = float(value)
                except ValueError:
                    st.error(f"Enter a valid number for {feature}.")
            else:
                patient_data[feature] = value.strip()

    submitted = st.form_submit_button("Predict Disorder")

if submitted:
    input_df = pd.DataFrame([patient_data])
    input_df = input_df.reindex(columns=feature_columns)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.success("Prediction completed!")

    st.subheader("Predicted Genetic Disorder")
    st.write(f"### {prediction}")

    st.subheader("Prediction Probabilities")

    for label, probability in zip(model.classes_, probabilities):
        st.write(f"**{label}**")
        st.progress(float(probability))
        st.write(f"{probability * 100:.2f}%")

    st.info("Educational prototype only. Not a medical diagnosis.")

st.caption("Genetic Disorder Risk Classification System")

# MODEL PERFORMANCE DASHBOARD

st.divider()

st.header("📊 Model Performance Dashboard")

with open("model_evaluation.json", "r") as file:
    evaluation = json.load(file)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{evaluation['accuracy']*100:.2f}%")
col2.metric("Precision", f"{evaluation['precision']*100:.2f}%")
col3.metric("Recall", f"{evaluation['recall']*100:.2f}%")
col4.metric("F1 Score", f"{evaluation['f1_score']*100:.2f}%")

st.subheader("Confusion Matrix")

class_names = evaluation["class_names"]

confusion_df = pd.DataFrame(
    evaluation["confusion_matrix"],
    index=class_names,
    columns=class_names
)

st.dataframe(confusion_df)

st.caption(
    "Evaluation metrics are calculated using the test dataset."
)

# EDA VISUALIZATIONS

st.divider()
st.header("📊 Exploratory Data Analysis")

st.subheader("1. Genetic Disorder Distribution")

st.image(
    "disorder_distribution.png",
    caption="Distribution of Genetic Disorder Categories",
    use_container_width=True
)

st.subheader("2. Patient Age Distribution")

st.image(
    "patient_age_distribution.png",
    caption="Distribution of Patient Ages",
    use_container_width=True
)




