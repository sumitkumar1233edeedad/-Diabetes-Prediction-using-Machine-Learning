import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# =====================================
# LOAD MODEL + SCALER (SAFE LOAD)
# =====================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("Diabetes.pkl")
        scaler = joblib.load("scaled_diabetes.pkl")
        return model, scaler
    except Exception as e:
        st.error("❌ Model or scaler file not found.")
        st.stop()

model, scaler = load_assets()

# =====================================
# HEADER
# =====================================
st.title("🩺 Diabetes Risk Prediction System")
st.markdown("### AI-Powered Clinical Risk Screening Tool")

st.info("Enter patient clinical parameters in the sidebar and click **Run Prediction**")

# =====================================
# SIDEBAR INPUTS
# =====================================
st.sidebar.header("🧾 Patient Parameters")

preg = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose Level", 50, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 40, 140, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.slider("Age", 1, 100, 30)

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

# =====================================
# MAIN LAYOUT
# =====================================
col1, col2 = st.columns([1, 1])

# ---------- INPUT SUMMARY ----------
with col1:
    st.subheader("📋 Input Summary")

    df_input = pd.DataFrame(input_data, columns=[
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DPF","Age"
    ])

    st.dataframe(df_input, use_container_width=True)

    st.caption("Verify values before running prediction")

# =====================================
# PREDICTION SECTION
# =====================================
if st.button("🔍 Run Prediction", use_container_width=True):

    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    with col2:
        st.subheader("📊 Prediction Result")

        result_box = st.container(border=True)

        with result_box:
            if pred == 1:
                st.error("⚠️ High Diabetes Risk Detected")
            else:
                st.success("✅ Low Diabetes Risk")

            st.metric(
                label="Risk Probability",
                value=f"{prob:.2%}"
            )

            st.write("Confidence Level")
            st.progress(float(prob))

            if prob > 0.7:
                st.warning("Recommendation: Clinical follow-up advised")
            elif prob > 0.4:
                st.info("Moderate risk — lifestyle monitoring suggested")
            else:
                st.success("Low predicted clinical risk")

# =====================================
# FEATURE IMPORTANCE
# =====================================
st.divider()
st.subheader("📈 Model Feature Importance")

if hasattr(model, "feature_importances_"):

    feat_names = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DPF","Age"
    ]

    importance = pd.Series(
        model.feature_importances_,
        index=feat_names
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8,5))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Feature Impact on Model Decision")
    ax.set_xlabel("Importance Score")

    st.pyplot(fig)

else:
    st.info("Model does not support feature importance")

# =====================================
# FOOTER
# =====================================
st.divider()

st.markdown("""
<div style='text-align:center; font-size:14px; color:gray;'>
Built with Streamlit • ML Diabetes Prediction Project<br>
Developed by <b>Vanshuu</b><br>
<a href="https://github.com/sumitkumar1233edeedad" target="_blank">
GitHub Profile
</a>
</div>
""", unsafe_allow_html=True)
