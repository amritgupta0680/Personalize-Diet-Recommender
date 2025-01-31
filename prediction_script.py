import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Instead of importing 'mean_squared_error' directly, import 'MSE'
from tensorflow.keras.losses import MSE  # type: ignore # Import MSE

# Load the trained model, providing 'mse' as a custom object
# Use 'MSE' instead of 'mean_squared_error' here as well
model = tf.keras.models.load_model("data_model.h5", custom_objects={'mse': MSE})

# Load dataset to ensure consistent preprocessing logic
data = pd.read_csv("synthetic_diet_dataset.csv")

# Recreate label encoders for categorical variables
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

# Recreate the scaler for numerical features
scaler = StandardScaler()
data[["Weight", "Height", "Age"]] = scaler.fit_transform(data[["Weight", "Height", "Age"]])

# Prediction function
def predict_diet_plan(weight, height, age, gender, goal, activity_level):
    # Encode categorical inputs
    gender_encoded = le_gender.transform([gender])[0]
    goal_encoded = le_goal.transform([goal])[0]
    activity_encoded = le_activity.transform([activity_level])[0]

    # Scale numerical inputs
    scaled_features = scaler.transform([[weight, height, age]])

    # Combine all inputs
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

