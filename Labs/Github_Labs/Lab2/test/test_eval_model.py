import os
import json
import pytest
from joblib import load
from subprocess import run


@pytest.mark.order(1)
def test_evaluate_model_creates_metrics(tmp_path):
    """
    Ensures that evaluate_model.py produces a valid metrics JSON file
    after running with a recent timestamp.
    """
    # --- Locate latest model and test data ---
    model_files = sorted(
        [os.path.join("models", f)
         for f in os.listdir("models") if f.endswith(".joblib")],
        key=os.path.getmtime,
        reverse=True
    )
    assert model_files, "No trained model found in 'models/' directory."

    latest_model = model_files[0]
    # Extract timestamp from model name
    timestamp = latest_model.split("_")[1]

    assert os.path.exists(
        "data/test_split.joblib"), "Missing test_split.joblib file."

    # --- Run the evaluate_model script ---
    result = run(
        ["python", "src/evaluate_model.py", "--timestamp", timestamp],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"evaluate_model.py exited with error: {result.stderr}"

    # --- Verify metrics file created ---
    metrics_path = f"metrics/{timestamp}_metrics.json"
    assert os.path.exists(
        metrics_path), f"Expected metrics file {metrics_path} not found."

    # --- Load and validate metrics content ---
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    for key in ["accuracy", "f1_score"]:
        assert key in metrics, f"Missing key '{key}' in metrics JSON."
        assert isinstance(metrics[key], float), f"{key} should be a float."
        assert 0.0 <= metrics[key] <= 1.0, f"{key} should be between 0 and 1."

    print(f"✅ Verified metrics file: {metrics_path}")


@pytest.mark.order(2)
def test_evaluate_model_uses_existing_model():
    """
    Ensures evaluate_model does not fail when rerun on existing model files.
    """
    model_files = [f for f in os.listdir("models") if f.endswith(".joblib")]
    assert model_files, "No model file found in models/."

    timestamp = model_files[-1].split("_")[1]
    metrics_path = f"metrics/{timestamp}_metrics.json"

    # Run evaluation again
    result = run(
        ["python", "src/evaluate_model.py", "--timestamp", timestamp],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"evaluate_model failed on re-run: {result.stderr}"
    assert os.path.exists(
        metrics_path), f"Metrics file not created after re-run: {metrics_path}"
