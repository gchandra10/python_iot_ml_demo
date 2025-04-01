import pandas as pd
from prophet import Prophet
import glob, joblib

# Load all CSVs
all_files = glob.glob("data/*.csv")
df_list = [pd.read_csv(f, header=None) for f in all_files]

# Combine into one dataframe
df = pd.concat(df_list, ignore_index=True)

df.columns = ["date", "hour", "temp", "dwpt", "rhum", "prcp", "snow", 
"wdir", "wspd", "wpgt", "pres", "tsun", "coco"]

# Combine date + hour into datetime
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str).str.zfill(2) + ':00:00')

selected_features = ["datetime", "temp", "rhum", "pres", "wspd", "dwpt", "prcp", "snow"]
df_prophet = df[selected_features].copy()
df_prophet = df_prophet.rename(columns={"datetime": "ds", "temp": "y"})

df_prophet[['rhum', 'pres', 'wspd', 'dwpt', 'prcp', 'snow']] = df_prophet[['rhum', 'pres', 'wspd', 'dwpt', 'prcp', 'snow']].fillna(0)
df_prophet = df_prophet.sort_values("ds")

# Fit model
model = Prophet()

# Add regressors for other features
model.add_regressor('rhum')
model.add_regressor('pres')
model.add_regressor('wspd')
model.add_regressor('dwpt')
model.add_regressor('prcp')
model.add_regressor('snow')

model.fit(df_prophet)

# Save model
joblib.dump(model, "model/prophet_model.pkl")
print("Model saved to prophet_model.pkl")