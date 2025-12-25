import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score

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

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y,
    test_size=config['test_size'],
    random_state=config['random_state'],
    stratify=y
)

# Train model
model = LogisticRegression(random_state=config['random_state'])
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, config['path'])

# Optional: test accuracy (can keep for your own checking)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")
