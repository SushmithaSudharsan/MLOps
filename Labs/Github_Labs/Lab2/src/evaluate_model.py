import os
import json
import argparse
import joblib
import pickle
from sklearn.metrics import f1_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default=os.getcwd(),
                        help="Base directory for models, metrics, and data (default: GitHub workspace root)")
    args = parser.parse_args()

    timestamp = args.timestamp
    base_dir = args.base_dir
    print(f"[evaluate_model.py] Timestamp: {timestamp}")
    print(f"[evaluate_model.py] Base directory: {base_dir}")

    # Paths
    model_dir = os.path.join(base_dir, "models")
    metrics_dir = os.path.join(base_dir, "metrics")
    data_dir = os.path.join(base_dir, "data")

    # Ensure directories exist
    os.makedirs(metrics_dir, exist_ok=True)

    # Load model
    model_filename = os.path.join(
        model_dir, f"model_{timestamp}_rf_model.joblib")
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")
    clf = joblib.load(model_filename)
    print(f"[evaluate_model.py] Loaded model: {model_filename}")

    # Load dataset
    X_file = os.path.join(data_dir, "data.pickle")
    y_file = os.path.join(data_dir, "target.pickle")
    if not os.path.exists(X_file) or not os.path.exists(y_file):
        raise FileNotFoundError(
            "Training data not found. Make sure train_model.py has run."
        )

    with open(X_file, 'rb') as f:
        X = pickle.load(f)
    with open(y_file, 'rb') as f:
        y = pickle.load(f)

    print(
        f"[evaluate_model.py] Loaded dataset: X shape={X.shape}, y shape={y.shape}")

    # Predict and calculate metrics
    y_pred = clf.predict(X)
    f1 = f1_score(y, y_pred, average='macro')
    metrics = {"F1_Score": f1}
    print(f"[evaluate_model.py] F1 Score: {f1}")

    # Save metrics
    metrics_filename = os.path.join(metrics_dir, f"{timestamp}_metrics.json")
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[evaluate_model.py] Metrics saved: {metrics_filename}")
