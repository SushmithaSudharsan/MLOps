import os
import pickle
import argparse
from joblib import dump
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default=os.getcwd(),
                        help="Base directory for models, data, and mlruns (default: GitHub workspace root)")
    args = parser.parse_args()

    timestamp = args.timestamp
    base_dir = args.base_dir
    print(f"[train_model.py] Timestamp: {timestamp}")
    print(f"[train_model.py] Base directory: {base_dir}")

    # Paths
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
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

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
        model_path = os.path.join(model_dir, model_filename)
        dump(clf, model_path)
        print(f"[train_model.py] Model saved: {model_path}")
