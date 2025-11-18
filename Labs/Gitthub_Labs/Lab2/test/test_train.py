import os
from joblib import load
import pytest


def test_model_artifact_exists():
    assert os.path.exists("models"), "models directory should exist"
    files = os.listdir("models")
    assert any(f.endswith(".joblib") for f in files), "No model file found"


def test_model_predicts_correct_shape():
    model_files = [f for f in os.listdir("models") if f.endswith(".joblib")]
    if not model_files:
        pytest.skip("No model to test yet.")
    model = load(os.path.join("models", model_files[0]))
    X_test, y_test = load("data/test_split.joblib")
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)
