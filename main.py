import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Загрузка модели
model = joblib.load('churn_model.pkl')

st.set_page_config(page_title="Gym Churn Prediction", layout="wide")
st.title("Прогноз оттока клиента фитнес-клуба")

# --- Ввод данных пользователем ---
age = st.number_input("Age", min_value=10, max_value=100, value=28)
gender = st.selectbox("Gender", ["Male", "Female"])
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0)
height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.78)
max_bpm = st.number_input("Max_BPM", min_value=80, max_value=220, value=180)
avg_bpm = st.number_input("Avg_BPM", min_value=60, max_value=200, value=120)
resting_bpm = st.number_input("Resting_BPM", min_value=40, max_value=120, value=70)
session_duration = st.number_input("Session_Duration (hours)", min_value=0.3, max_value=5.0, value=0.7)
calories_burned = st.number_input("Calories_Burned", min_value=50.0, max_value=3000.0, value=250.0)
workout_type = st.selectbox("Workout_Type", ["Yoga", "HIIT", "Cardio", "Strength"])
fat_percentage = st.number_input("Fat_Percentage", min_value=5.0, max_value=50.0, value=18.0)
water_intake = st.number_input("Water_Intake (liters)", min_value=0.5, max_value=10.0, value=2.0)
workout_frequency = st.number_input("Workout_Frequency (days/week)", min_value=1, max_value=7, value=2)
experience_level = st.selectbox("Experience_Level", [1, 2, 3], index=0)

# --- Кодирование категориальных признаков ---
# Важно: используйте те же правила кодирования, что и при обучении!
gender_map = {"Male": 1, "Female": 0}  # Проверьте, какой класс какому числу соответствует в вашем LabelEncoder
workout_type_map = {"Yoga": 0, "HIIT": 1, "Cardio": 2, "Strength": 3}  # Проверьте порядок!

gender_encoded = gender_map[gender]
workout_type_encoded = workout_type_map[workout_type]
bmi = weight / (height ** 2)
calories_per_hour = calories_burned / session_duration

# Определяем возрастную группу и кодируем её
age_bins = [0, 25, 35, 45, 100]
age_labels = ['18-25', '26-35', '36-45', '46+']
age_group = pd.cut([age], bins=age_bins, labels=age_labels)[0]
age_group_map = {'18-25': 0, '26-35': 1, '36-45': 2, '46+': 3}  # Проверьте соответствие с вашим LabelEncoder!
age_group_encoded = age_group_map[str(age_group)]

# --- Формируем DataFrame с нужными признаками ---
input_data = pd.DataFrame([[
    age,
    gender_encoded,
    weight,
    height,
    max_bpm,
    avg_bpm,
    resting_bpm,
    session_duration,
    calories_burned,
    workout_type_encoded,
    fat_percentage,
    water_intake,
    workout_frequency,
    experience_level,
    bmi,
    calories_per_hour,
    age_group_encoded
]], columns=[
    'Age',
    'Gender_encoded',
    'Weight (kg)',
    'Height (m)',
    'Max_BPM',
    'Avg_BPM',
    'Resting_BPM',
    'Session_Duration (hours)',
    'Calories_Burned',
    'Workout_Type_encoded',
    'Fat_Percentage',
    'Water_Intake (liters)',
    'Workout_Frequency (days/week)',
    'Experience_Level',
    'BMI',
    'Calories_per_hour',
    'Age_group_encoded'
])

if st.button("Сделать прогноз"):
    prediction = model.predict(input_data)
    st.write("Результат прогноза (1 - уйдёт, 0 - останется):", int(prediction[0]))
