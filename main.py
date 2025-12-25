
# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit
import yaml
import pandas as pd
df = pd.read_csv("parkinsons.csv")
df.head()
df = df.dropna()

import seaborn as sns
sns.pairplot(df, hue="status")
X = df[["MDVP:Fo(Hz)", "HNR"]]
y = df["status"]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled.head(), y.head()

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train.shape, X_val.shape, y_train.shape, y_val.shape

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

