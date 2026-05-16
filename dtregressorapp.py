import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# ---------------- TITLE ---------------- #

st.title("Used Car Price Prediction")

st.write("Predict estimated used car price in Indian Rupees")

# ---------------- LOAD DATASET ---------------- #

df = pd.read_csv("CarPrice_Assignment.csv")

# Create Brand column
df["Brand"] = df["CarName"].apply(lambda x: x.split()[0])

# Create fake Year column
np.random.seed(42)

df["Year"] = np.random.randint(2010, 2024, size=len(df))

# Keep required columns
df = df[
    [
        "Brand",
        "fueltype",
        "Year",
        "price"
    ]
]

# ---------------- ENCODING ---------------- #

encoders = {}

for col in ["Brand", "fueltype"]:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

# ---------------- FEATURES & TARGET ---------------- #

X = df.drop("price", axis=1)

y = df["price"]

# ---------------- SPLIT DATA ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL ---------------- #

model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- USER INPUTS ---------------- #

st.subheader("Enter Car Details")

# Brand
brand = st.selectbox(
    "Select Car Brand",
    encoders["Brand"].classes_
)

# Fuel Type
fuel = st.selectbox(
    "Fuel Type",
    encoders["fueltype"].classes_
)

# Year
year = st.slider(
    "Manufacturing Year",
    2010,
    2023,
    2019
)

# ---------------- PREDICT BUTTON ---------------- #

if st.button("Predict Price"):

    # Create input dataframe
    input_data = pd.DataFrame([[
        encoders["Brand"].transform([brand])[0],
        encoders["fueltype"].transform([fuel])[0],
        year
    ]], columns=X.columns)

    # Prediction
    prediction = model.predict(input_data)

    predicted_price = int(prediction[0])

    # Convert USD-like value to INR
    price_in_rupees = predicted_price * 83

    # Convert to Lakhs
    price_in_lakhs = price_in_rupees / 100000

    # Final Output
    st.success(
        f"Estimated Car Price: ₹{price_in_lakhs:.1f} Lakhs"
    )