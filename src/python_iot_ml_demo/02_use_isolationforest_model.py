import pandas as pd
import joblib
import glob
from tabulate import tabulate

# Load the trained model
model = joblib.load("model/iso_forest_model.pkl")

# Simulate new sensor data (e.g., temp only)
# In a real-world scenario, this would be replaced with actual sensor data
new_data = pd.DataFrame({
    "temp": [87.5, 32.8, 3.2, -0.9, 2.5] 
})

# Convert to numeric in case it comes as string
# errors="coerce" will convert non-numeric values to NaN
# This is important to ensure that the model can process the data correctly
new_data["temp"] = pd.to_numeric(new_data["temp"], errors="coerce")

# Drop rows with missing values
new_data = new_data.dropna()

# Predict anomalies: -1 = anomaly, 1 = normal
new_data["anomaly"] = model.predict(new_data[["temp"]])

# Filter anomalies
anomalies = new_data[new_data["anomaly"] == -1]

# Show in pretty table
print(tabulate(anomalies.head(10), headers='keys', tablefmt='pretty', showindex=False))
