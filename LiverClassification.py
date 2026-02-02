# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 17:33:10 2026

@author: adiya
"""
import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="Liver Disease Prediction", layout="wide")
st.title("🩺 Liver Disease Prediction System (Gradient Boosting)")

# ---------------------------------
# Load Gradient Boosting Models
# ---------------------------------
BASE_DIR = os.getcwd()

try:
    gb_stage1 = pickle.load(open(os.path.join(BASE_DIR, "gb_stage1_model.pkl"), "rb"))
    gb_stage2 = pickle.load(open(os.path.join(BASE_DIR, "gb_stage2_model.pkl"), "rb"))
    le2 = pickle.load(open(os.path.join(BASE_DIR, "gb_stage2_encoder.pkl"), "rb"))
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")
    st.stop()

# ---------------------------------
# User Input Section
# ---------------------------------
st.subheader("📋 Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=40)

sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0  # must match training encoding

albumin = st.number_input("Albumin", value=4.0)
alkaline_phosphatase = st.number_input("Alkaline Phosphatase", value=100.0)
alanine_aminotransferase = st.number_input("ALT (Alanine Aminotransferase)", value=30.0)
aspartate_aminotransferase = st.number_input("AST (Aspartate Aminotransferase)", value=30.0)
bilirubin = st.number_input("Bilirubin", value=1.0)
cholinesterase = st.number_input("Cholinesterase", value=7000.0)
cholesterol = st.number_input("Cholesterol", value=180.0)
creatinina = st.number_input("Creatinine", value=1.0)
gamma_glutamyl_transferase = st.number_input("GGT (Gamma Glutamyl Transferase)", value=40.0)
protein = st.number_input("Total Protein", value=7.0)

# ---------------------------------
# Create input dataframe (ORDER MATTERS)
# ---------------------------------
user_input = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "albumin": albumin,
    "alkaline_phosphatase": alkaline_phosphatase,
    "alanine_aminotransferase": alanine_aminotransferase,
    "aspartate_aminotransferase": aspartate_aminotransferase,
    "bilirubin": bilirubin,
    "cholinesterase": cholinesterase,
    "cholesterol": cholesterol,
    "creatinina": creatinina,
    "gamma_glutamyl_transferase": gamma_glutamyl_transferase,
    "protein": protein
}])

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("🔍 Predict"):

    # ========= STAGE 1 =========
    # Disease vs No Disease
    stage1_pred = gb_stage1.predict(user_input)[0]
    stage1_probs = gb_stage1.predict_proba(user_input)[0]

    st.subheader("🧪 Stage-1: Disease Screening")

    st.write(
        f"No Disease: {stage1_probs[0]*100:.2f}% | "
        f"Disease: {stage1_probs[1]*100:.2f}%"
    )

    if stage1_pred == 0:
        st.success("🟢 No Liver Disease Detected")

    else:
        st.error("🔴 Liver Disease Detected")

        # ========= STAGE 2 =========
        # Disease Type Prediction
        stage2_probs = gb_stage2.predict_proba(user_input)[0]
        classes = le2.classes_

        st.subheader("🩺 Stage-2: Disease Type Probability")

        for cls, prob in zip(classes, stage2_probs):
            st.write(f"**{cls.replace('_',' ').title()}** : {prob*100:.2f}%")

        # Bar chart
        fig, ax = plt.subplots()
        ax.bar(
            [c.replace('_', ' ').title() for c in classes],
            stage2_probs * 100
        )
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)
        st.pyplot(fig)

        # Final disease
        final_idx = stage2_probs.argmax()
        st.warning(
            f"🩺 Most Likely Condition: "
            f"{classes[final_idx].replace('_',' ').title()}"
        )

# ---------------------------------
# Disclaimer
# ---------------------------------
st.info(
    "⚠️ This system is intended for educational and screening purposes only. "
    "It should not be considered a medical diagnosis."
)
