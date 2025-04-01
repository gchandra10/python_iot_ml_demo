import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import joblib

# Load actual CSV
actual_df = pd.read_csv("test_data/kttno_2024.csv", header=None)
actual_df.columns = ["date", "hour", "temp", "dwpt", "rhum", "prcp", "snow", 
"wdir", "wspd", "wpgt", "pres", "tsun", "coco"]

# Create datetime
actual_df['datetime'] = pd.to_datetime(actual_df['date'] + ' ' + actual_df['hour'].astype(str).str.zfill(2) + ':00:00')

# Prepare actuals (for comparison only)
actuals = actual_df[['datetime', 'temp']].rename(columns={"datetime": "ds", "temp": "actual"})
actuals = actuals.dropna().sort_values("ds")

# Load the trained model
model = joblib.load("model/prophet_model.pkl")

# Prepare regressors + ds
future = actual_df[['datetime', 'rhum', 'pres', 'wspd', 'dwpt', 'prcp', 'snow']].copy()
future = future.rename(columns={"datetime": "ds"})
future = future.sort_values("ds")
future['snow'] = future['snow'].fillna(0)

# forward fill and backward fill to handle NaNs
future = future.ffill().bfill()

# Ensure no NaNs
if future.isnull().values.any():
    print("Remaining NaNs:\n", future.isna().sum())
    raise ValueError("Future DataFrame still has NaNs. Aborting.")

# Predict using external regressors
forecast = model.predict(future)

# Merge forecast with actuals
compare_df = forecast[['ds', 'yhat']].merge(actuals, on='ds', how='inner')

# Save to CSV
compare_df.to_csv("predicted_vs_actual.csv", index=False)

# Print head
print(compare_df.head())

print("\n\n")
# Print summary statistics
print("=== Summary Statistics ===")
print(f"Count: {compare_df['actual'].count()}")

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_true = compare_df['actual']
y_pred = compare_df['yhat']

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MAPE is not defined when y_true is 0
# To avoid division by zero, we can filter out those cases
# Only consider valid cases where y_true is not zero
# We can also set a threshold to avoid division by zero

# valid is a Boolean array (Pandas Series), not a single value.
valid = y_true.abs() > 0.1
mape = (abs((y_true[valid] - y_pred[valid]) / y_true[valid])).mean() * 100

# MAE = average of |actual - predicted|
# Lower MAE is better
print(f"MAE (Mean Absolute Error): {mae:.2f}")

# RMSE = sqrt(mean((actual - predicted)^2))
# Lower RMSE is better
# RMSE is sensitive to outliers and is always Positive
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")

# MAPE = mean absolute percentage error
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

print("\n")

print("=== Interpretation of Errors ===")
print("If MAE and RMSE are close, there are no major outliers")
print("""
Mape < 10% - Highly Accurate
Mape < 20% - Accurate
Mape < 50% - Fairly Accurate
Mape > 50% - Poor Accuracy      
""")

############## Plotting

# Downsample for cleaner visualization if needed (e.g., every 12 hours)
sampled_df = compare_df[::12] 

plt.plot(sampled_df['ds'], sampled_df['actual'], label='Actual', linewidth=1)
plt.plot(sampled_df['ds'], sampled_df['yhat'], label='Predicted', linewidth=1)

plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Actual vs Predicted Temperature (Hourly Forecast)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


