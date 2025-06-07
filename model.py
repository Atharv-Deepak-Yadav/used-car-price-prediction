import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("used_cars_data.csv")

# Drop rows with missing target or critical features
df = df.dropna(subset=['Price', 'Name', 'Year', 'Kilometers_Driven', 
                       'Fuel_Type', 'Transmission', 'Owner_Type', 'Engine'])

# Extract numeric part from Engine column (e.g., "1248 CC" → 1248)
df['Engine_CC'] = df['Engine'].astype(str).str.extract('(\d+)').astype(float)

# Encode car name
label_encoder = LabelEncoder()
df['car_name_encoded'] = label_encoder.fit_transform(df['Name'])

# Encode categorical features using mapping
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3, "LPG": 4, "Hybrid": 5}
trans_map = {"Manual": 0, "Automatic": 1}
owner_map = {"First": 0, "Second": 1, "Third": 2, "Fourth & Above": 3, "Test Drive": 4}

df['fuel_encoded'] = df['Fuel_Type'].map(fuel_map).fillna(0)
df['trans_encoded'] = df['Transmission'].map(trans_map).fillna(0)
df['owner_encoded'] = df['Owner_Type'].astype(str).str.extract(r'(\w+)')[0].map(owner_map).fillna(0)

# Define features and target variable
features = ['car_name_encoded', 'Year', 'Kilometers_Driven', 'fuel_encoded',
            'trans_encoded', 'owner_encoded', 'Engine_CC']
X = df[features]
y = df['Price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and label encoder
joblib.dump(model, "car_price_model_small.pkl")
joblib.dump(label_encoder, "label_encoder_small.pkl")

print("✅ Model trained and saved as car_price_model.pkl and label_encoder.pkl")
