import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

data_apath = "data/processed/training_data.csv"
df = pd.read_csv(data_apath)
features = ['stop_lat', 'stop_lon','arrival_min','stop_sequence']
X = df[features].dropna()
# Change this line in train_model.py:
y = df.loc[X.index, 'scheduled_travel_time']  # Was 'scheduled_arrival_time'
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

print("Training delay model")
model = RandomForestRegressor(n_estimators=20, max_depth=10,random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Model mean absolute error: {mae:.2f} minutes")
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/delay_model.pkl")
print("Model saved to models/delay_model.pkl")
