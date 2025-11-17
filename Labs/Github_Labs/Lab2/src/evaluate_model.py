import os
import json
import argparse
import joblib
import pickle
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
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(metrics_dir, exist_ok=True)

    # Load model
    model_filename = os.path.join(
        model_dir, f"model_{timestamp}_rf_model.joblib")
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")
    clf = joblib.load(model_filename)
    print(f"[evaluate_model.py] Loaded model: {model_filename}")

    # Load the dataset saved during training
    X_file = os.path.join(data_dir, "data.pickle")
    y_file = os.path.join(data_dir, "target.pickle")

    if not os.path.exists(X_file) or not os.path.exists(y_file):
        raise FileNotFoundError(
            "Training data not found. Make sure train_model.py has run.")

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
