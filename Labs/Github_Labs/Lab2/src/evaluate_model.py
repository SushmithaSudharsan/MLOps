import pickle
import os
import json
import joblib
import argparse
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp
    print(f"[evaluate_model.py] Timestamp: {timestamp}")

    # Define folder paths
    model_dir = "Github_Labs/Lab2/models"
    metrics_dir = "Github_Labs/Lab2/metrics"

    os.makedirs(metrics_dir, exist_ok=True)

    # Load the model
    model_filename = f"{model_dir}/model_{timestamp}_dt_model.joblib"
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")

    model = joblib.load(model_filename)
    print(f"[evaluate_model.py] Loaded model: {model_filename}")

    # Generate synthetic data for evaluation
    try:
        X, y = make_classification(
            n_samples=1000,  # fixed size for evaluation
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
    except Exception as e:
        raise ValueError(f"Failed to generate synthetic data: {e}")

    # Make predictions and calculate metrics
    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred)
    metrics = {"F1_Score": f1}
    print(f"[evaluate_model.py] F1 Score: {f1}")

    # Save metrics to JSON file
    metrics_filename = f"{metrics_dir}/{timestamp}_metrics.json"
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[evaluate_model.py] Metrics saved: {metrics_filename}")
