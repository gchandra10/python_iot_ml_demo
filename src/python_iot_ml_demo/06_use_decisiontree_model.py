import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load trained model
model = joblib.load("model/machine_state_dt_model.pkl")

# New sensor reading
sensor_data = pd.DataFrame([{"temp": 80.2, "vibration": 1.2, "pressure": 101.5}])

# Predict state
prediction = model.predict(sensor_data)
print("Predicted State:", prediction[0])
