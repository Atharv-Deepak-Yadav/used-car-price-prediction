# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("car_price_model_small.pkl")
label_encoder = joblib.load("label_encoder_small.pkl")

# Load dataset (used for "Dataset" page)
df = pd.read_csv("used_cars_data.csv")

# Clean dataset to show in dataset section
df_clean = df.dropna().copy()
df_clean['Engine'] = df_clean['Engine'].astype(str).str.extract('(\d+)').astype(float)

# Maps
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3, "LPG": 4}
trans_map = {"Manual": 0, "Automatic": 1}
owner_map = {"First Owner": 0, "Second Owner": 1, "Third Owner": 2,
             "Fourth & Above Owner": 3, "Test Drive Car": 4}

# Set page config
st.set_page_config(page_title="Used Car Price Predictor", layout="wide")

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Predict"])

# ------------------- PAGE: HOME ------------------- #
if page == "Home":
    st.title("🚗 Used Car Price Predictor")
    st.markdown("""
    Welcome to the **Used Car Price Predictor** app!  
    This app uses machine learning to estimate the resale price of used cars based on input features like:
    
    - Car model
    - Year of manufacture
    - Kilometers driven
    - Fuel type
    - Transmission
    - Ownership history
    - Engine capacity
    
    Navigate using the sidebar to:
    - View the dataset used
    - Predict car prices interactively
    """)

# ------------------- PAGE: DATASET ------------------- #
elif page == "Dataset":
    st.title("📊 Dataset Preview")
    st.markdown("Below is the dataset used to train the model:")
    st.dataframe(df_clean.reset_index(drop=True))

# ------------------- PAGE: PREDICT ------------------- #
elif page == "Predict":
    st.title("🤖 Predict Used Car Price")

    available_cars = sorted(label_encoder.classes_)
    car_name = st.selectbox("Select Car Model", available_cars)
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
    fuel = st.selectbox("Fuel Type", list(fuel_map.keys()))
    trans = st.selectbox("Transmission", list(trans_map.keys()))
    owner = st.selectbox("Owner Type", list(owner_map.keys()))
    engine = st.number_input("Engine (in CC)", min_value=500, max_value=10000, value=1200)

    if st.button("Predict Price"):
        name_encoded = label_encoder.transform([car_name])[0]
        fuel_encoded = fuel_map[fuel]
        trans_encoded = trans_map[trans]
        owner_encoded = owner_map[owner]

        features = np.array([[name_encoded, year, km_driven, fuel_encoded,
                              trans_encoded, owner_encoded, engine]])
        prediction = model.predict(features)[0]
        st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} lakhs")
