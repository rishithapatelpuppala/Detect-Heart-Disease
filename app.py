import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Detector", layout="centered")
st.title("❤️ Heart Disease Prediction App")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 250)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])

if st.button("Predict"):
    sex = 1 if sex == "Male" else 0
    cp_dict = {"Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal Pain": 3, "Asymptomatic": 4}
    restecg_dict = {"Normal": 0, "ST-T wave abnormality": 1, "Left Ventricular Hypertrophy": 2}
    slope_dict = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    data = pd.DataFrame([[age, sex, cp_dict[cp], trestbps, chol, fbs,
                          restecg_dict[restecg], thalach, exang, oldpeak,
                          slope_dict[slope]]],
                        columns=['age', 'sex', 'chestpaintype', 'restingbps',
                                 'cholesterol', 'fastingbloodsugar', 'restingecg',
                                 'maxheartrate', 'exerciseangina', 'oldpeak', 'stslope'])

    data = pd.get_dummies(data, columns=['chestpaintype', 'restingecg', 'stslope'])

    expected_cols = ['chestpaintype_1', 'chestpaintype_2', 'chestpaintype_3', 'chestpaintype_4',
                     'restingecg_0', 'restingecg_1', 'restingecg_2',
                     'stslope_0', 'stslope_1', 'stslope_2', 'stslope_3']

    for col in expected_cols:
        if col not in data.columns:
            data[col] = 0

    data = data[['age', 'sex', 'restingbps', 'cholesterol', 'fastingbloodsugar',
                 'maxheartrate', 'exerciseangina', 'oldpeak'] + expected_cols]

    scaler = joblib.load("scaler.pkl")
    model = joblib.load("heart_disease_model.pkl")

    numeric_cols = ['age', 'restingbps', 'cholesterol', 'maxheartrate', 'oldpeak']
    data[numeric_cols] = scaler.transform(data[numeric_cols])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("🔴 High Risk: The patient has heart disease.")
    else:
        st.success("🟢 Low Risk: The patient does not have heart disease.")
