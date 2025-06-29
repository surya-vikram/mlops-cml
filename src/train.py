# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load data
df = pd.read_csv('data/iris.csv')

# 2. Split
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train for 200 iterations
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Save
joblib.dump(model, 'artifacts/model.joblib')