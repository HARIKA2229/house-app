import streamlit as st
import numpy as np
import pickle

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

st.title("🏡 House Price Prediction")
st.write("Enter the your dream house to predict price.")

# ----------------------------------
# Load model and scaler
# ----------------------------------
@st.cache_resource
def load_artifacts():
    with open("new_rfmodel_pkl", "rb") as f:
        model = pickle.load(f)

    with open("new_scalar_pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_artifacts()

# ----------------------------------
# Feature inputs
# ----------------------------------
feature_inputs = {
    'Square_footage': st.number_input('Square_footage', min_value=503.00, value=1030),
    'Num_Bedrooms': st.number_input('Num_Bedrooms', min_value=1, value=1),
    'Num_Bathrooms': st.number_input('Citric Acid', min_value=0.0, value=0.0),
    'Year_Built': st.number_input('Year_Built', min_value=0.0, value=1.9),
    'Lot_Size': st.number_input('Lot_Size', min_value=0.0, value=0.076),
    'Garage_Size': st.number_input('Garage_Size', min_value=0.0, value=11.0),
    'tNeighborhood_Quality': st.number_input('Neighborhood_Quality', min_value=0.0, value=34.0),
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




