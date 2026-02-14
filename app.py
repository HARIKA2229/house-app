import streamlit as st
import numpy as np
import pickle
import os
import base64
from pathlib import Path


# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="centered"
) 
def set_bg(image_file):
    img_path = Path(image_file)

    # If image not found, stop and show error
    if not img_path.exists():
        st.error(f"❌ Background image not found: {image_file}")
        st.stop()

    encoded = base64.b64encode(img_path.read_bytes()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Your image name here
set_bg("house1.png") 


st.title("🏡 House Price Prediction")
st.write("Enter the your dream house to predict price.")

# ----------------------------------
# Load model and scaler
# ----------------------------------
@st.cache_resource
def load_artifacts():
    with open("new_rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("new_scalar.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_artifacts()

# ----------------------------------
# Feature inputs
# ----------------------------------
feature_inputs = {
    'Square_footage': st.number_input('Square_footage', min_value=503, value=1030),
    'Num_Bedrooms': st.number_input('Num_Bedrooms', min_value=1, value=1),
    'Num_Bathrooms': st.number_input('Num_Bathrooms', min_value=1, value=1),
    'Year_Built': st.number_input('Year_Built', min_value=1950, value=1990),
    'Lot_Size': st.number_input('Lot_Size', min_value=0.0, value=3.4),
    'Garage_Size': st.number_input('Garage_Size', min_value=0.0, value=1.0),
    'tNeighborhood_Quality': st.number_input('Neighborhood_Quality', min_value=1.0, value=8.0),
}

# Maintain correct feature order
feature_names = list(feature_inputs.keys())
input_values = [feature_inputs[f] for f in feature_names]

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("House price"):
    input_array = np.array(input_values).reshape(1, -1)

    # Scale input
    scaled_input = scaler.transform(input_array)

    # Predict
    prediction = model.predict(scaled_input)


    st.success(f"🏡 Predicted House price: **{int(prediction[0])}**")












