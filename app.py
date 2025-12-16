import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="ML Prediction App", layout="centered")

st.title("🔮 ML Prediction App")
st.write("This Streamlit app is connected to a GitHub repository")

# Load model
@st.cache_resource
def load_model():
    with open("battery_predict.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# User inputs
st.subheader("Enter Input Values")

feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)

    st.success(f"✅ Prediction Result: {prediction[0]}")
