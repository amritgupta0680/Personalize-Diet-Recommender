import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset (for consistent preprocessing)
data = pd.read_csv("synthetic_diet_dataset.csv")

# Load LabelEncoders used during training
le_gender = LabelEncoder()
le_goal = LabelEncoder()
le_activity = LabelEncoder()
le_diet_plan = LabelEncoder()
le_food_items = LabelEncoder()

data["Gender"] = le_gender.fit_transform(data["Gender"])
data["Goal"] = le_goal.fit_transform(data["Goal"])
data["Activity_Level"] = le_activity.fit_transform(data["Activity_Level"])
data["Diet_Plan"] = le_diet_plan.fit_transform(data["Diet_Plan"])
data["Food_Items"] = le_food_items.fit_transform(data["Food_Items"])

# Load Scaler used during training
scaler = StandardScaler()
data[["Weight", "Height", "Age"]] = scaler.fit_transform(data[["Weight", "Height", "Age"]])

# Load the trained model
model = tf.keras.models.load_model("data_model.h5")

# Streamlit App Title
st.title("Personalized Diet Recommendation System")

# Sidebar Input Fields
st.sidebar.header("User Inputs")
weight = st.sidebar.slider("Weight (kg)", 40, 150, step=1, value=70)
height = st.sidebar.slider("Height (cm)", 140, 210, step=1, value=170)
age = st.sidebar.slider("Age (years)", 18, 80, step=1, value=25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
goal = st.sidebar.selectbox("Goal", ["Fat Loss", "Muscle Gain", "Maintenance"])
activity_level = st.sidebar.selectbox("Activity Level", ["Sedentary", "Moderate", "Active"])

# Prediction Function
def predict_diet_plan(weight, height, age, gender, goal, activity_level):
    # Encode categorical inputs
    gender_encoded = le_gender.transform([gender])[0]
    goal_encoded = le_goal.transform([goal])[0]
    activity_encoded = le_activity.transform([activity_level])[0]

    # Scale numerical inputs
    scaled_features = scaler.transform([[weight, height, age]])

    # Prepare input array for model
    inputs = np.array([[scaled_features[0][0], scaled_features[0][1], scaled_features[0][2], gender_encoded, goal_encoded, activity_encoded]])

    # Make predictions
    predictions = model.predict(inputs)

    # Decode predictions
    water_intake = round(predictions[0][0], 2)
    protein_intake = round(predictions[0][1], 2)
    diet_plan = le_diet_plan.inverse_transform([int(round(predictions[0][2]))])[0]
    food_items = le_food_items.inverse_transform([int(round(predictions[0][3]))])[0]

    return {
        "Water Intake (L/day)": water_intake,
        "Protein Intake (g/day)": protein_intake,
        "Diet Plan": diet_plan,
        "Food Items": food_items
    }

# Predict Button
if st.sidebar.button("Get Recommendations"):
    # Get predictions
    result = predict_diet_plan(weight, height, age, gender, goal, activity_level)
    
    # Display results
    st.subheader("Your Personalized Diet Plan")
    st.write(f"💧 **Water Intake:** {result['Water Intake (L/day)']} L/day")
    st.write(f"🥩 **Protein Intake:** {result['Protein Intake (g/day)']} g/day")
    st.write(f"📋 **Diet Plan:** {result['Diet Plan']}")
    st.write(f"🍽️ **Recommended Food Items:** {result['Food Items']}")
