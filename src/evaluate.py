# src/evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

def evaluate(threshold=0.9):
    df = pd.read_csv('data/iris.csv')
    X = df.drop('species', axis=1)
    y = df['species']
    model = joblib.load('model.joblib')
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f'Accuracy: {acc:.3f}')
    return acc >= threshold

if __name__ == '__main__':
    success = evaluate()
    if not success:
        raise SystemExit('Accuracy below threshold!')
