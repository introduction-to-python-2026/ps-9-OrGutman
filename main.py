import yaml
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load dataset
df = pd.read_csv("parkinsons.csv")

# Select features and target
X = df[config['features']]
y = df[config['target']]

# Scale features
scaler = MinMaxScaler(feature_range=(config['scaler_min'], config['scaler_max']))
X_scaled = scaler.fit_transform(X)

# Train the model on all data
model = LogisticRegression(random_state=config['random_state'], max_iter=1000)
model.fit(X_scaled, y)

# Save the trained model
joblib.dump(model, config['path'])
