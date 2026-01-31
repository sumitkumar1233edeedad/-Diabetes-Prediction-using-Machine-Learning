import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# -------------------------
# Load Assets
# -------------------------
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------
# Header
# -------------------------
st.title("🩺 Diabetes Risk Prediction System")
st.markdown("AI-powered clinical risk screening tool")

# -------------------------
# Sidebar Input Form
# -------------------------
st.sidebar.header("Patient Parameters")

preg = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose Level", 50, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 40, 140, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.slider("Age", 1, 100, 30)

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

# -------------------------
# Layout Columns
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Input Summary")

    df_input = pd.DataFrame(input_data, columns=[
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DPF","Age"
    ])

    st.dataframe(df_input, use_container_width=True)

# -------------------------
# Prediction
# -------------------------
if st.button("🔍 Run Prediction", use_container_width=True):

    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    with col2:
        st.subheader("📊 Prediction Result")

        if pred == 1:
            st.error("⚠️ High Diabetes Risk")
        else:
            st.success("✅ Low Diabetes Risk")

        st.metric("Risk Probability", f"{prob:.2%}")

        st.progress(float(prob))

# -------------------------
# Feature Importance
# -------------------------
st.divider()
st.subheader("📈 Model Feature Importance")

if hasattr(model, "feature_importances_"):
    feat_names = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DPF","Age"
    ]

    imp = pd.Series(model.feature_importances_, index=feat_names).sort_values()

    fig, ax = plt.subplots()
    imp.plot(kind="barh", ax=ax)
    st.pyplot(fig)

# -------------------------
# Footer
# -------------------------
st.caption("Built with Streamlit • ML Diabetes Project")
