import mlflow
import datetime
import os
import pickle
from joblib import dump
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()

    timestamp = args.timestamp
    print(f"[train_model.py] Timestamp: {timestamp}")

    # Load Wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    print(f"[train_model.py] Dataset shape: {X.shape}")

    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)

    # Save dataset
    with open('data/data.pickle', 'wb') as f:
        pickle.dump(X, f)
    with open('data/target.pickle', 'wb') as f:
        pickle.dump(y, f)

    # MLflow setup
    mlflow.set_tracking_uri("./mlruns")
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
        dump(clf, f"models/{model_filename}")
        print(f"[train_model.py] Model saved: models/{model_filename}")
