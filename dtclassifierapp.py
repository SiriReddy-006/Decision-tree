import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------------- PAGE TITLE ---------------- #

st.title("Heart Disease Risk Predictor")

st.write("Check your possible heart disease risk using simple health details")

# ---------------- LOAD DATASET ---------------- #

df = pd.read_csv("heart.csv")

# Features and target
X = df.drop("target", axis=1)

y = df["target"]

# ---------------- SPLIT DATA ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL ---------------- #

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- USER INPUTS ---------------- #

st.subheader("Enter Your Health Details")

# Age
age = st.slider(
    "Age",
    20,
    80,
    45
)

# Gender
sex = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

sex_value = 1 if sex == "Male" else 0

# Chest Pain
cp_options = {
    "No Chest Pain": 0,
    "Mild Chest Pain": 1,
    "Moderate Chest Pain": 2,
    "Severe Chest Pain": 3
}

cp_selected = st.selectbox(
    "Chest Pain Level",
    list(cp_options.keys())
)

cp = cp_options[cp_selected]

# Blood Pressure
trestbps = st.slider(
    "Blood Pressure",
    80,
    200,
    120,
    help="Normal blood pressure is around 120"
)

# Cholesterol
chol = st.slider(
    "Cholesterol Level",
    100,
    600,
    220,
    help="Higher cholesterol may increase heart disease risk"
)

# Blood Sugar
fbs = st.selectbox(
    "High Blood Sugar",
    ["No", "Yes"]
)

fbs_value = 1 if fbs == "Yes" else 0

# Heart Test Result
restecg_options = {
    "Normal": 0,
    "Minor Issue": 1,
    "Major Issue": 2
}

restecg_selected = st.selectbox(
    "Heart Test Result",
    list(restecg_options.keys())
)

restecg = restecg_options[restecg_selected]

# Heart Rate
thalach = st.slider(
    "Maximum Heart Rate",
    60,
    220,
    150
)

# Chest Pain During Exercise
exang = st.selectbox(
    "Chest Pain During Exercise",
    ["No", "Yes"]
)

exang_value = 1 if exang == "Yes" else 0

# Stress Level During Exercise
oldpeak = st.slider(
    "Stress Level During Exercise",
    0.0,
    6.0,
    1.0,
    step=0.1,
    help="Higher values may indicate heart stress"
)

# Heart Activity
slope_options = {
    "Normal": 0,
    "Moderate Issue": 1,
    "Serious Issue": 2
}

slope_selected = st.selectbox(
    "Heart Activity During Exercise",
    list(slope_options.keys())
)

slope = slope_options[slope_selected]

# Blocked Blood Vessels
ca = st.selectbox(
    "Blocked Blood Vessels",
    [0, 1, 2, 3]
)

# Blood Flow Condition
thal_options = {
    "Normal": 1,
    "Fixed Problem": 2,
    "Reversible Problem": 3
}

thal_selected = st.selectbox(
    "Blood Flow Condition",
    list(thal_options.keys())
)

thal = thal_options[thal_selected]

# ---------------- PREDICT BUTTON ---------------- #

if st.button("Check Heart Disease Risk"):

    # Create dataframe
    input_data = pd.DataFrame([[
        age,
        sex_value,
        cp,
        trestbps,
        chol,
        fbs_value,
        restecg,
        thalach,
        exang_value,
        oldpeak,
        slope,
        ca,
        thal
    ]], columns=X.columns)

    # Prediction probability
    probability = model.predict_proba(input_data)[0][1]

    risk_percent = probability * 100

    # ---------------- OUTPUT ---------------- #

    st.subheader("Prediction Result")

    if risk_percent >= 70:

        st.error(
            f"High Risk of Heart Disease ({risk_percent:.1f}%)"
        )

    elif risk_percent >= 40:

        st.warning(
            f"Moderate Risk of Heart Disease ({risk_percent:.1f}%)"
        )

    else:

        st.success(
            f"Low Risk of Heart Disease ({risk_percent:.1f}%)"
        )