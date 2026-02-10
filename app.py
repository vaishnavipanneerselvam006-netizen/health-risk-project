import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Health Risk Prediction",
    layout="wide"
)

st.title("🏥 Health Risk Prediction App")
st.write("Predict health risk and understand feature impact using ML + SHAP.")

# -------------------------------------------------
# Load model & preprocessors
# -------------------------------------------------
@st.cache_data
def load_artifacts():
    model = joblib.load("models/random_forest_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, preprocessor, label_encoder

model, preprocessor, label_encoder = load_artifacts()

# -------------------------------------------------
# Sidebar – User input
# -------------------------------------------------
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 0, 120, 30)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 22.0)
sleep_hours = st.sidebar.number_input("Sleep Hours", 0, 24, 7)
stress_level = st.sidebar.slider("Stress Level", 0, 10, 5)
exercise_days = st.sidebar.slider("Exercise Days per Week", 0, 7, 3)
diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
pcos = st.sidebar.selectbox("PCOS", ["No", "Yes"])
diet_type = st.sidebar.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# -------------------------------------------------
# Input dataframe
# -------------------------------------------------
input_df = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "sleep_hours": sleep_hours,
    "stress_level": stress_level,
    "exercise_days": exercise_days,
    "diabetes": 1 if diabetes == "Yes" else 0,
    "pcos": 1 if pcos == "Yes" else 0,
    "diet_type": diet_type,
    "gender": gender
}])

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
X_processed = preprocessor.transform(input_df)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
pred_probs = model.predict_proba(X_processed)[0]
pred_class = np.argmax(pred_probs)
pred_label = label_encoder.inverse_transform([pred_class])[0]

st.subheader("🔍 Prediction Result")
st.write(f"**Predicted Health Risk:** `{pred_label}`")

st.write("**Class Probabilities:**")
for i, cls in enumerate(label_encoder.classes_):
    st.write(f"- {cls}: {pred_probs[i]:.2f}")

# -------------------------------------------------
# SHAP Explanation (SHAP ≥ 0.40 SAFE)
# -------------------------------------------------
st.subheader("🧠 Explainable AI (SHAP)")

explainer = shap.TreeExplainer(model)
shap_exp = explainer(X_processed)  # shap.Explanation object

# Select predicted class
shap_values = shap_exp[:, :, pred_class]
expected_value = explainer.expected_value[pred_class]

feature_names = preprocessor.get_feature_names_out()

# -------------------------------------------------
# Global Feature Importance
# -------------------------------------------------
st.write("### 📊 Global Feature Importance")

fig_summary, ax = plt.subplots()
shap.summary_plot(
    shap_values.values,
    X_processed,
    feature_names=feature_names,
    show=False
)
st.pyplot(fig_summary)
plt.close(fig_summary)

# -------------------------------------------------
# Local Explanation (Waterfall)
# -------------------------------------------------
st.write("### 🧩 Local Explanation for This Patient")

fig_waterfall, ax = plt.subplots()
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values.values[0],
        base_values=expected_value,
        data=X_processed[0],
        feature_names=feature_names
    ),
    show=False
)
st.pyplot(fig_waterfall)
plt.close(fig_waterfall)
