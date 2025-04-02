import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Simulated training dataset
# Replace this with your real IoT data
data = pd.DataFrame({
    "temp": [70.1, 85.3, 90.0, 60.2, 78.5, 95.2],
    "vibration": [0.5, 1.4, 1.8, 0.3, 1.0, 2.1],
    "pressure": [101.2, 99.5, 98.1, 102.3, 100.0, 97.5],
    "state": ["normal", "warning", "fault", "normal", "warning", "fault"]
})

# Features and label
X = data[["temp", "vibration", "pressure"]]
y = data["state"]

# Train the model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, "model/machine_state_dt_model.pkl")
print("Model saved as machine_state_dt_model.pkl")
