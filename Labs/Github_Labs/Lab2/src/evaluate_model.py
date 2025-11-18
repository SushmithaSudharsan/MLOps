import argparse
import json
import os
from joblib import load
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Evaluating model for timestamp {timestamp}")

    # === Load model & data ===
    model_path = f"models/model_{timestamp}_rf_model.joblib"
    X_test, y_test = load("data/test_split.joblib")
    model = load(model_path)

    # === Evaluate ===
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred, average='macro'), 4)
    }

    # === Save metrics JSON ===
    os.makedirs("metrics", exist_ok=True)
    metrics_filename = f"metrics/{timestamp}_metrics.json"
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Metrics saved → {metrics_filename}")
