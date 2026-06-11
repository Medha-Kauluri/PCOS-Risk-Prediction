# streamlit run app/streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="PCOS Risk Predictor", layout="centered")

st.title("PCOS Risk Predictor")
st.markdown(
    "This is a **screening tool for educational use only**. "
    "It estimates the probability of PCOS based on your inputs."
)

MODEL_PATH = Path("models/pcos_model.joblib")

REQUIRED_FEATURES = [
    "follicle_r",
    "follicle_l",
    "hirsutism",
    "amh",
    "bmi",
    "lh",
    "weight",
    "age",
    "irregular_cycle",
    "height",
    "acne",
]


@st.cache_resource
def load_model(path=MODEL_PATH):
    if not path.exists():
        return None
    return joblib.load(path)


model = load_model()

if model is None:
    st.error("Trained model not found. Run train_pcos.py first.")
else:
    st.success("Model loaded.")


st.sidebar.header("Input patient data")

age = st.sidebar.number_input("Age (yrs)", min_value=10, max_value=60, value=28)
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=60.0)
height = st.sidebar.number_input("Height (cm)", min_value=130.0, max_value=200.0, value=160.0)

bmi = weight / ((height / 100.0) ** 2)
st.sidebar.write(f"Calculated BMI: **{bmi:.1f}**")

amh = st.sidebar.number_input("AMH (ng/mL)", min_value=0.0, max_value=20.0, value=0.0)
lh = st.sidebar.number_input("LH (mIU/mL)", min_value=0.0, max_value=40.0, value=0.0)

follicle_r = st.sidebar.number_input("Follicle count – Right ovary", min_value=0, max_value=40, value=10)
follicle_l = st.sidebar.number_input("Follicle count – Left ovary", min_value=0, max_value=40, value=10)

hirsutism_opt = st.sidebar.selectbox("Hirsutism (excess hair growth)?", ["No", "Yes"])
acne_opt = st.sidebar.selectbox("Acne/pimples?", ["No", "Yes"])
irregular_opt = st.sidebar.selectbox("Irregular menstrual cycle?", ["No", "Yes"])

hirsutism = 1 if hirsutism_opt == "Yes" else 0
acne = 1 if acne_opt == "Yes" else 0
irregular_cycle = 1 if irregular_opt == "Yes" else 0

if st.button("Predict PCOS risk"):

    if model is None:
        st.error("Model is not loaded.")
    else:
        
        row = {
            "follicle_r": follicle_r,
            "follicle_l": follicle_l,
            "hirsutism": hirsutism,
            "amh": amh if amh != 0 else np.nan,
            "bmi": bmi,
            "lh": lh if lh != 0 else np.nan,
            "weight": weight,
            "age": age,
            "irregular_cycle": irregular_cycle,
            "height": height,
            "acne": acne,
        }

        X = pd.DataFrame([row])

        # Make sure columns match training exactly
        missing_cols = set(REQUIRED_FEATURES) - set(X.columns)
        if missing_cols:
            st.error(f"Internal error: app is missing columns: {missing_cols}")
        else:
            try:
                proba = model.predict_proba(X)[:, 1][0]
                st.metric("Estimated PCOS Probability", f"{proba * 100:.1f}%")

                if proba < 0.3:
                    st.success("Low risk pattern (screening only).")
                elif proba < 0.7:
                    st.warning("Moderate risk. Consider clinical evaluation.")
                else:
                    st.error("High-risk pattern. Clinical assessment is strongly recommended.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
