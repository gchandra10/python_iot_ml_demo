import pandas as pd
import glob
from sklearn.ensemble import IsolationForest
import joblib
import matplotlib.pyplot as plt
from tabulate import tabulate

# Load all CSVs
all_files = glob.glob("data/*.csv")
df_list = [pd.read_csv(f, header=None) for f in all_files]

# Combine into one dataframe
df = pd.concat(df_list, ignore_index=True)

# Assign column names
df.columns = [
    "date", "hour", "temp", "dwpt", "rhum", "prcp", "snow", 
    "wdir", "wspd", "wpgt", "pres", "tsun", "coco"
]

# Convert to numeric (coerce errors for missing values)
for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with too many NaNs or fill with mean (optional)
df_clean = df.dropna(subset=["temp", "rhum", "prcp", "wspd", "pres"]).copy()

# Features for anomaly detection
#features = df_clean[["temp", "rhum", "prcp", "wspd", "pres"]]
features = df_clean[["temp"]]

# EDA (Exploratory Data Analysis) for 'temp' before model training
print("=== EDA for 'temp' ===")
print(f"Count: {df_clean['temp'].count()}")
print(f"Min: {df_clean['temp'].min()}")
print(f"Max: {df_clean['temp'].max()}")
print(f"Mean: {df_clean['temp'].mean():.2f}")
print(f"Std: {df_clean['temp'].std():.2f}")
print(f"25%: {df_clean['temp'].quantile(0.25)}")
print(f"50% (Median): {df_clean['temp'].median()}")
print(f"75%: {df_clean['temp'].quantile(0.75)}")

######### Plotting

plt.hist(df_clean["temp"].dropna(), bins=50, edgecolor='black')
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

######### Model Training
# n_estimators: Number of base estimators in the ensemble
# contamination: The proportion of outliers in the data set
# random_state: Controls the randomness of the estimator

model = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
model.fit(features)

# Save model
joblib.dump(model, "model/iso_forest_model.pkl")
print("Model saved to iso_forest_model.pkl")

# Predict anomalies: -1 = anomaly, 1 = normal
df_clean["anomaly"] = model.predict(features)

# Filter anomalies
df_clean = df_clean.drop(columns=["dwpt", "rhum", "prcp", "snow", "wdir","pres", "wspd","wpgt", "tsun", "coco"])
anomalies = df_clean[df_clean["anomaly"] == -1]

# Show in pretty table
print(tabulate(anomalies.head(), headers='keys', tablefmt='pretty', showindex=False))