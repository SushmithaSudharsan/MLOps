import os
import joblib
import argparse
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp from GitHub Actions")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory for models and data")
    args = parser.parse_args()

    timestamp = args.timestamp
    base_dir = args.base_dir
    print(f"Evaluating model for timestamp {timestamp}")
    print(f"Base directory: {base_dir}")

    # Paths
    model_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Load model
    model_filename = os.path.join(
        model_dir, f"model_{timestamp}_rf_model.joblib")
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")
    clf = joblib.load(model_filename)
    print(f"Loaded model: {model_filename}")

    # Load test split
    test_split_file = os.path.join(data_dir, "test_split.joblib")
    if not os.path.exists(test_split_file):
        raise FileNotFoundError(
            f"Test split file not found: {test_split_file}")
    X_test, y_test = joblib.load(test_split_file)
    print(
        f"Loaded test split: X_test shape={X_test.shape}, y_test shape={y_test.shape}")

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')  # Important for multiclass

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4)
    }

    # Save metrics
    metrics_filename = os.path.join(metrics_dir, f"{timestamp}_metrics.json")
    import json
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved: {metrics_filename}")
    print(
        f"✅ Evaluation complete: Accuracy={metrics['accuracy']}, F1 Score={metrics['f1_score']}")
