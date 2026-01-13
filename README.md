# 🥗 AI-Based Nutrition Recommendation System

An end-to-end **machine learning–powered nutrition recommendation system** that predicts personalized daily **water intake**, **protein requirements**, and **diet plans** based on user health attributes and fitness goals. The system is deployed as an interactive **Streamlit web application** for real-time recommendations.

---

## 🚀 Project Overview

This project demonstrates the complete **machine learning lifecycle**:

* Data understanding and preprocessing
* Model training and evaluation
* Rule-based post-processing for domain accuracy
* Deployment as a user-facing web application

The model focuses on **regression-based nutrition prediction**, enhanced with **nutrition science rules** to ensure realistic and safe recommendations.

---

## 🎯 Features

* Predicts **daily water intake (liters)**
* Predicts **daily protein requirements (grams)**
* Recommends a suitable **diet plan category**
* Provides **vegetarian or non-vegetarian food suggestions**
* Interactive UI built with **Streamlit**

---

## 🧠 Machine Learning Details

### Model Type

* **Multi-output regression neural network**
* Built using **TensorFlow / Keras**

### Input Features

* Weight (kg)
* Height (cm)
* Age
* Gender
* Fitness Goal (Fat Loss / Muscle Gain / Maintenance)
* Activity Level (Sedentary / Moderate / Active)

### Model Outputs

* Daily Water Intake (L)
* Daily Protein Requirement (g) – baseline
* Diet Plan Category (encoded)

---

## 📊 Model Performance

The model was evaluated using regression metrics:

* **Mean Absolute Error (MAE): 0.61**
* **R² Score: 0.84**

These results indicate strong predictive performance for continuous nutrition targets.

---

## 🧩 Hybrid Recommendation Logic

To ensure physiologically valid recommendations, the system uses a **hybrid approach**:

* **Machine Learning Model** → Predicts baseline nutrition values and diet plan
* **Rule-Based Logic** → Refines protein intake based on:

  * User fitness goal (muscle gain, fat loss, maintenance)
  * Dietary preference (vegetarian / non-vegetarian)

This combination improves real-world reliability beyond raw ML predictions.

---

## 🥦 Food Recommendation Logic

* The ML model predicts a **diet plan category**
* Food items are **retrieved from the dataset** based on:

  * Predicted diet plan
  * User diet preference (vegetarian / non-vegetarian)

> Note: Food items are **recommended via data lookup**, not directly predicted by the model.

---

## 📁 Dataset Information

* The dataset is a **synthetic nutrition dataset**
* Created using **nutrition science guidelines and rule-based logic**
* Avoids privacy issues associated with real medical data
* Includes both numerical and categorical features

This approach allows controlled experimentation and end-to-end ML deployment.

---

## 🖥️ Application Architecture

```
User Input
   ↓
Preprocessing (Encoding + Scaling)
   ↓
Trained Neural Network
   ↓
Nutrition Predictions
   ↓
Rule-Based Protein Refinement
   ↓
Diet Plan & Food Recommendations
```

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **Scikit-learn**
* **Pandas, NumPy**
* **Streamlit**

---

## ▶️ How to Run the Project

1. Clone the repository

   ```bash
   git clone <repository-url>
   cd nutrition-recommendation-system
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app

   ```bash
   streamlit run app.py
   ```

---

## 📌 Disclaimer

This project is intended for **educational and demonstration purposes only**. The recommendations are not a substitute for professional medical or dietary advice.

---

## 👤 Author

**Amrit Gupta**
B.E. in Artificial Intelligence & Data Science

---

## ⭐ Future Improvements

* Automated dataset generation script
* Food portion optimization based on protein targets
* Integration with real-world nutrition databases
* User authentication and progress tracking

---

⭐ If you find this project useful, feel free to star the repository!
