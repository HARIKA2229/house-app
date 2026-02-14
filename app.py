import streamlit as st
import numpy as np
import pickle
import os
import base64


# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Add your background image here
add_bg_from_local("house.png") 

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









