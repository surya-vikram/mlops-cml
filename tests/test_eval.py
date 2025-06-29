# tests/test_eval.py

import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Inline the evaluate logic so no external imports are needed
def evaluate(threshold=0.9):
    # load the CSV from the data folder
    df = pd.read_csv('data/iris.csv')

    # split features / target
    X = df.drop('species', axis=1)
    y = df['species']

    # load the pre-trained model artifact
    model = joblib.load('artifacts/model.joblib')

    # run predictions & compute accuracy
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f'Accuracy: {acc:.3f}')

    # return whether we met the threshold
    return acc >= threshold

# The actual pytest test
def test_model_accuracy():
    # you can tweak the threshold here if needed
    assert evaluate(threshold=0.9)
