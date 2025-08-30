import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

model = tf.keras.models.load_model('Breast_CanCer_Train_Model.keras')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        h1, h2, h3, h4, p, label, div {
            color: #ffffff !important;
        }
        div.stButton > button {
            background-color: #00c0f2;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 18px;
        }
        div.stButton > button:hover {
            background-color: #008bb5;
            color: #ffffff;
        }
        .stSuccess {
            background-color: #1b4332;
            color: #95d5b2;
        }
        .section-box {
            background: #1a1d24;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
        }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Breast Cancer Prediction App")
st.write("Fill out the patient’s medical features below and predict whether the tumor is **Benign** or **Malignant**.")

mean_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
                 "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
                 "mean fractal dimension"]

error_features = ["radius error", "texture error", "perimeter error", "area error",
                  "smoothness error", "compactness error", "concavity error",
                  "concave points error", "symmetry error", "fractal dimension error"]

worst_features = ["worst radius", "worst texture", "worst perimeter", "worst area",
                  "worst smoothness", "worst compactness", "worst concavity",
                  "worst concave points", "worst symmetry", "worst fractal dimension"]

input_values = []

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("📌 Mean Features")
col1, col2, col3 = st.columns(3)
for i, feature in enumerate(mean_features):
    if i % 3 == 0:
        value = col1.number_input(feature, value=0.0)
    elif i % 3 == 1:
        value = col2.number_input(feature, value=0.0)
    else:
        value = col3.number_input(feature, value=0.0)
    input_values.append(value)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("📌 Error Features")
col1, col2, col3 = st.columns(3)
for i, feature in enumerate(error_features):
    if i % 3 == 0:
        value = col1.number_input(feature, value=0.0)
    elif i % 3 == 1:
        value = col2.number_input(feature, value=0.0)
    else:
        value = col3.number_input(feature, value=0.0)
    input_values.append(value)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("📌 Worst Features")
col1, col2, col3 = st.columns(3)
for i, feature in enumerate(worst_features):
    if i % 3 == 0:
        value = col1.number_input(feature, value=0.0)
    elif i % 3 == 1:
        value = col2.number_input(feature, value=0.0)
    else:
        value = col3.number_input(feature, value=0.0)
    input_values.append(value)
st.markdown('</div>', unsafe_allow_html=True)

input_array = np.array([input_values])
input_scaled = scaler.transform(input_array)

if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)
    result = "⚠️ Malignant (Cancer Detected)" if prediction[0][0] > 0.5 else "✅ Benign (No Cancer)"
    
    if "Malignant" in result:
        st.error(f"Prediction: {result}")
    else:
        st.success(f"Prediction: {result}")
