from src.evaluate import evaluate

def test_model_accuracy():
    assert evaluate(threshold=0.9)