import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
data = pd.read_csv("data/updated_diet_dataset_dynamic.csv")

# Encoding models
le_gender, le_goal, le_activity, le_diet_plan = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()
data["Gender"], data["Goal"], data["Activity_Level"], data["Diet_Plan"] = \
    le_gender.fit_transform(data["Gender"]), le_goal.fit_transform(data["Goal"]), \
    le_activity.fit_transform(data["Activity_Level"]), le_diet_plan.fit_transform(data["Diet_Plan"])

# Scaling model
scaler = StandardScaler()
data[["Weight", "Height", "Age"]] = scaler.fit_transform(data[["Weight", "Height", "Age"]])

# Load trained model
model = tf.keras.models.load_model("models/data_model (1).h5")

st.title("🥗 Personalized Diet Recommendation")

st.sidebar.header("Enter Your Details")
weight = st.sidebar.slider("Weight (kg)", 40, 150, 70)
height = st.sidebar.slider("Height (cm)", 140, 210, 170)
age = st.sidebar.slider("Age (years)", 18, 80, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
goal = st.sidebar.selectbox("Goal", ["Fat Loss", "Muscle Gain", "Maintenance"])
activity_level = st.sidebar.selectbox("Activity Level", ["Sedentary", "Moderate", "Active"])
diet_preference = st.sidebar.selectbox("Diet Preference", ["Vegetarian", "Non-Vegetarian"])

# Protein content per 100g
food_protein_data = {
    "Paneer": 18,
    "Dal": 25,
    "Brown Rice": 2.6,
    "Almonds": 21,
    "Chicken Breast": 31,
    "Mixed Vegetables": 2
}

def adjust_food_quantities(protein_needed, diet_type):
    """Adjusts food item quantities to match required protein intake."""
    food_items = {
        "Vegetarian": {"Paneer": 50, "Dal": 100, "Brown Rice": 150, "Almonds": 5, "Mixed Vegetables": 200},
        "Non-Vegetarian": {"Chicken Breast": 100, "Dal": 100, "Brown Rice": 150, "Almonds": 5, "Mixed Vegetables": 200}
    }

    selected_foods = food_items[diet_type]
    total_protein = sum((food_protein_data[item] * (amount / 100)) for item, amount in selected_foods.items())

    while total_protein < protein_needed:
        for item in selected_foods.keys():
            selected_foods[item] += 10  # Increase quantity by 10g
            total_protein = sum((food_protein_data[item] * (amount / 100)) for item, amount in selected_foods.items())
            if total_protein >= protein_needed:
                break

    return ", ".join([f"{amount}g {item}" for item, amount in selected_foods.items()])

def predict_diet(weight, height, age, gender, goal, activity_level, diet_preference):
    gender_encoded = le_gender.transform([gender])[0]
    goal_encoded = le_goal.transform([goal])[0]
    activity_encoded = le_activity.transform([activity_level])[0]

    scaled_input = scaler.transform([[weight, height, age]])
    input_data = np.array([[scaled_input[0][0], scaled_input[0][1], scaled_input[0][2], gender_encoded, goal_encoded, activity_encoded]])

    prediction = model.predict(input_data)
    protein_required = round(prediction[0][1], 2)

    food_items = adjust_food_quantities(protein_required, diet_preference)

    return {
        "Water Intake (L/day)": round(prediction[0][0], 2),
        "Protein Intake (g/day)": protein_required,
        "Diet Plan": le_diet_plan.inverse_transform([int(round(prediction[0][2]))])[0],
        "Food Items": food_items
    }

if st.sidebar.button("Get Recommendations"):
    result = predict_diet(weight, height, age, gender, goal, activity_level, diet_preference)
    st.write(f"💧 **Water Intake:** {result['Water Intake (L/day)']} L/day")
    st.write(f"🥩 **Protein Intake:** {result['Protein Intake (g/day)']} g/day")
    st.write(f"📋 **Diet Plan:** {result['Diet Plan']}")
    st.write(f"🍽️ **Recommended Food Items:** {result['Food Items']}")
