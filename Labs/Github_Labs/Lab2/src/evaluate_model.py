import os
import joblib
import argparse
import json
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":

    # -----------------------------
    # 1️⃣ Parse arguments
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp used in model filename")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory where models/data/metrics are stored")
    args = parser.parse_args()

    timestamp = args.timestamp
    base_dir = args.base_dir

    print(f"Evaluating model for timestamp → {timestamp}")
    print(f"Base directory → {base_dir}")

    # -----------------------------
    # 2️⃣ Construct paths
    # -----------------------------
    model_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    metrics_dir = os.path.join(base_dir, "metrics")

    os.makedirs(metrics_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"model_{timestamp}_rf_model.joblib")
    test_split_path = os.path.join(data_dir, "test_split.joblib")
    metrics_output_path = os.path.join(
        metrics_dir, f"{timestamp}_metrics.json")

    # -----------------------------
    # 3️⃣ Load model
    # -----------------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    clf = joblib.load(model_path)
    print(f"Loaded model → {model_path}")

    # -----------------------------
    # 4️⃣ Load test split
    # -----------------------------
    if not os.path.exists(test_split_path):
        raise FileNotFoundError(f"❌ Test data not found: {test_split_path}")

    X_test, y_test = joblib.load(test_split_path)
    print(f"Loaded test split → X_test={X_test.shape}, y_test={y_test.shape}")

    # -----------------------------
    # 5️⃣ Predict and evaluate
    # -----------------------------
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")  # IMPORTANT for multiclass

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4)
    }

    # -----------------------------
    # 6️⃣ Save metrics
    # -----------------------------
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"📁 Metrics saved → {metrics_output_path}")
    print(f"🎉 Evaluation Complete → Accuracy={accuracy:.4f}, F1={f1:.4f}")
