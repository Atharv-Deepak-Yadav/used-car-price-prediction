import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("car_price_model_small.pkl")
label_encoder = joblib.load("label_encoder_small.pkl")

# Load dataset
df = pd.read_csv("used_cars_data.csv")

# Clean dataset
df_clean = df.dropna().copy()
df_clean['Engine'] = df_clean['Engine'].astype(str).str.extract('(\d+)').astype(float)

# Check if 'Kms_Driven' exists
if 'Kms_Driven' not in df_clean.columns:
    st.error("⚠️ 'Kms_Driven' column missing in the dataset! Check data source.")
else:
    df_clean['Kms_Driven'] = df_clean['Kms_Driven'].astype(float)

# Mapping dictionaries
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3, "LPG": 4}
trans_map = {"Manual": 0, "Automatic": 1}
owner_map = {"First Owner": 0, "Second Owner": 1, "Third Owner": 2,
             "Fourth & Above Owner": 3, "Test Drive Car": 4}

# Streamlit UI setup
st.set_page_config(page_title="Used Car Price Predictor", layout="wide")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Predict"])

# ------------------- HOME PAGE ------------------- #
if page == "Home":
    st.title("🚗 Used Car Price Predictor")
    st.markdown("""
    This app estimates the resale price of used cars using machine learning.
    """)

    st.subheader("📊 Quick Dataset Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Engine (CC)", f"{df_clean['Engine'].mean():.0f}")
    col2.metric("Most Common Fuel", df_clean['Fuel_Type'].mode()[0])
    col3.metric("Average KM Driven", f"{df_clean.get('Kms_Driven', pd.Series([np.nan])).mean():.0f}")

    st.subheader("📉 Fuel Type Distribution")
    st.bar_chart(df_clean['Fuel_Type'].value_counts())

    st.subheader("⬇️ Download Dataset")
    st.download_button("Download CSV", data=df_clean.to_csv(index=False), file_name="cleaned_data.csv")

# ------------------- DATASET PAGE ------------------- #
elif page == "Dataset":
    st.title("📊 Dataset Preview")
    st.dataframe(df_clean.reset_index(drop=True))

# ------------------- PREDICTION PAGE ------------------- #
elif page == "Predict":
    st.title("🤖 Predict Used Car Price")

    # Input options
    car_name = st.selectbox("Select Car Model", sorted(label_encoder.classes_))
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
    fuel = st.selectbox("Fuel Type", list(fuel_map.keys()))
    trans = st.selectbox("Transmission", list(trans_map.keys()))
    owner = st.selectbox("Owner Type", list(owner_map.keys()))
    engine = st.number_input("Engine (CC)", min_value=500, max_value=10000, value=1200)

    # Prediction Logic
    if st.button("Predict Price"):
        features = np.array([[label_encoder.transform([car_name])[0], year, km_driven,
                              fuel_map[fuel], trans_map[trans], owner_map[owner], engine]])
        prediction = model.predict(features)[0]
        st.success(f"💰 Estimated Price: ₹ {prediction:.2f} lakhs")
