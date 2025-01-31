import streamlit as st
from prediction_script import predict_diet_plan  # Assuming the prediction script is saved as prediction_script.py

# App Title
st.title("Personalized Diet Recommendation System")

# Input Fields
st.sidebar.header("User Inputs")
weight = st.sidebar.slider("Weight (kg)", 40, 150, step=1, value=70)
height = st.sidebar.slider("Height (cm)", 140, 210, step=1, value=170)
age = st.sidebar.slider("Age (years)", 18, 80, step=1, value=25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
goal = st.sidebar.selectbox("Goal", ["Fat Loss", "Muscle Gain", "Maintenance"])
activity_level = st.sidebar.selectbox("Activity Level", ["Sedentary", "Moderate", "Active"])

# Predict Button
if st.sidebar.button("Get Recommendations"):
    # Make predictions
    result = predict_diet_plan(weight, height, age, gender, goal, activity_level)
    
    # Display results
    st.subheader("Your Personalized Diet Plan")
    st.write(f"*Water Intake:* {result['Water Intake (L/day)']} L/day")
    st.write(f"*Protein Intake:* {result['Protein Intake (g/day)']} g/day")
    st.write(f"*Diet Plan:* {result['Diet Plan']}")
    st.write(f"*Recommended Food Items:* {result['Food Items']}")