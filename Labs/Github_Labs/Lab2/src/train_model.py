import os
import pickle
import argparse
from joblib import dump
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()

    timestamp = args.timestamp
    print(f"[train_model.py] Timestamp: {timestamp}")

    # Paths
    base_dir = "Labs/Github_Labs/Lab2"
    model_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    mlruns_dir = os.path.join(base_dir, "mlruns")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(mlruns_dir, exist_ok=True)

    # Load dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    print(f"[train_model.py] Dataset shape: {X.shape}")

    # Save dataset
    with open(os.path.join(data_dir, 'data.pickle'), 'wb') as f:
        pickle.dump(X, f)
    with open(os.path.join(data_dir, 'target.pickle'), 'wb') as f:
        pickle.dump(y, f)

    # MLflow setup
    mlflow.set_tracking_uri(mlruns_dir)
    experiment_name = f"Wine_{timestamp}"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(
            experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name="Wine Dataset"):
        mlflow.log_params({
            "dataset_shape": X.shape,
            "num_classes": len(set(y))
        })

        # Train model
        clf = RandomForestClassifier(random_state=0)
        clf.fit(X, y)
        y_pred = clf.predict(X)

        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, average='macro')
        })

        # Save model
        model_filename = f"model_{timestamp}_rf_model.joblib"
        dump(clf, os.path.join(model_dir, model_filename))
        print(
            f"[train_model.py] Model saved: {os.path.join(model_dir, model_filename)}")
