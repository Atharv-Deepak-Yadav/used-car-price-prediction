import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved model and label encoder
model = joblib.load("car_price_model.Z")
label_encoder = joblib.load("label_encoder.pkl")

# Maps (same as training)
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3, "LPG": 4, "Hybrid": 5}
trans_map = {"Manual": 0, "Automatic": 1}
owner_map = {"First Owner": 0, "Second Owner": 1, "Third Owner": 2,
             "Fourth & Above Owner": 3, "Test Drive Car": 4}

# UI
st.title("🚗 Predict Car Price")
st.subheader("Fill the details below:")

car_name = st.selectbox("Car Make/Model", label_encoder.classes_)
year = st.slider("Year of Manufacture", 2000, 2025, 2015)
km_driven = st.slider("Kilometers Driven (in KM)", 0, 300000, 50000)
engine_cc = st.number_input("Engine Capacity (CC)", min_value=500, max_value=6000, step=50, value=1200)

fuel_type = st.selectbox("Fuel Type", list(fuel_map.keys()))
transmission = st.selectbox("Transmission", list(trans_map.keys()))
owner_type = st.selectbox("Owner Type", list(owner_map.keys()))

if st.button("🚘 Predict Price"):
    input_data = pd.DataFrame([{
        'car_name_encoded': label_encoder.transform([car_name])[0],
        'Year': year,
        'Kilometers_Driven': km_driven,
        'fuel_encoded': fuel_map.get(fuel_type, 0),
        'trans_encoded': trans_map.get(transmission, 0),
        'owner_encoded': owner_map.get(owner_type, 0),
        'Engine_CC': engine_cc
    }])
    
    prediction = model.predict(input_data)[0]
    
    # Optional scaling (if model trained with Price in lakhs)
    estimated_price = round(prediction, 2)
    
    st.success(f"💰 Estimated Car Price: ₹ {estimated_price} Lakhs")
