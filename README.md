🥗 AI-Based Nutrition Recommendation System
📌 Overview

This project is an AI-powered nutrition recommendation system that provides personalized dietary guidance based on user health attributes and fitness goals.
The system predicts daily water intake, protein requirements, and diet plan categories using a machine learning regression model, and delivers food recommendations through an interactive Streamlit web application.

The project demonstrates a complete end-to-end machine learning pipeline, from data preprocessing and model training to real-time deployment.

🚀 Features

Personalized nutrition recommendations based on:

Weight, height, age

Gender

Activity level

Fitness goal (muscle gain, fat loss, maintenance)

Dietary preference (vegetarian / non-vegetarian)

Multi-output ML model to predict:

💧 Daily water intake

🥩 Daily protein requirement

📋 Diet plan category

Rule-based post-processing to ensure physiologically valid protein intake

Interactive and user-friendly web interface built with Streamlit

🧠 Machine Learning Approach
Model Type

Multi-output regression neural network

Built using TensorFlow (Keras)

Input Features

Weight (kg)

Height (cm)

Age

Gender (encoded)

Activity level (encoded)

Fitness goal (encoded)

Model Outputs

Water intake (liters/day)

Protein intake (grams/day – baseline)

Diet plan (encoded category)

📊 Model Performance

The model was evaluated using regression metrics:

Mean Absolute Error (MAE): 0.61

R² Score: 0.84

These metrics indicate strong predictive performance for continuous nutrition targets.

🧪 Dataset Description

The dataset is a synthetic, domain-driven nutrition dataset.

Created using nutrition science guidelines and rule-based logic.

Contains structured data with both numerical and categorical features.

Dataset Includes:

User attributes (weight, height, age, gender, activity level)

Nutrition targets (water intake, protein intake)

Diet plan categories

Vegetarian and non-vegetarian food recommendations

⚠️ Note: The dataset is synthetic and does not contain real user or medical data.

🧩 Protein Calculation Logic

The system uses a hybrid approach:

The ML model predicts a baseline protein value

Rule-based logic refines protein intake based on:

Fitness goal (muscle gain, fat loss, maintenance)

Dietary preference (vegetarian / non-vegetarian)

This ensures that protein recommendations align with established nutrition guidelines.

🍽️ Food Recommendation Logic

Food items are not predicted by the model

The predicted diet plan category is used to filter food options from the dataset

Food recommendations are displayed based on:

Diet plan

User’s dietary preference (veg / non-veg)

🖥️ Application Flow
User Input
   ↓
Data Encoding & Scaling
   ↓
Trained ML Model Prediction
   ↓
Protein Post-Processing Logic
   ↓
Diet Plan Mapping
   ↓
Food Recommendations Display

🛠️ Tech Stack

Python

TensorFlow / Keras

Scikit-learn

Pandas & NumPy

Streamlit

▶️ How to Run the Project

Clone the repository

git clone <repository-url>
cd nutrition-recommendation-system


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py

📌 Project Highlights

Demonstrates full ML lifecycle: data → model → evaluation → deployment

Combines machine learning predictions with domain-driven rules

Designed for extensibility (real-world datasets, nutrition APIs, portion control)

🔮 Future Improvements

Integrate real nutritional databases (e.g., USDA)

Add portion-based food recommendations

Automate synthetic dataset generation

Include calorie and micronutrient predictions

👤 Author

Amrit Gupta
B.E. in Artificial Intelligence & Data Science
📧 amritgupta0680@gmail.com

🔗 LinkedIn | GitHub
