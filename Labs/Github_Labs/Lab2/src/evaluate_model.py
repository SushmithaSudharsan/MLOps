import os
import json
import argparse
import joblib
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()

    timestamp = args.timestamp
    print(f"[evaluate_model.py] Timestamp: {timestamp}")

    # Paths
    base_dir = "Labs/Github_Labs/Lab2"
    model_dir = os.path.join(base_dir, "models")
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Load model
    model_filename = os.path.join(
        model_dir, f"model_{timestamp}_rf_model.joblib")
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")
    clf = joblib.load(model_filename)
    print(f"[evaluate_model.py] Loaded model: {model_filename}")

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0,
        shuffle=True,
    )
    print(
        f"[evaluate_model.py] Generated synthetic data: X shape={X.shape}, y shape={y.shape}")

    # Predict and calculate metrics
    y_pred = clf.predict(X)
    f1 = f1_score(y, y_pred)
    metrics = {"F1_Score": f1}
    print(f"[evaluate_model.py] F1 Score: {f1}")

    # Save metrics
    metrics_filename = os.path.join(metrics_dir, f"{timestamp}_metrics.json")
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[evaluate_model.py] Metrics saved: {metrics_filename}")
