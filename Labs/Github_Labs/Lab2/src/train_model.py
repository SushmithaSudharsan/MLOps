import mlflow
import datetime
import os
import pickle
from joblib import dump
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score
import sys
from sklearn.ensemble import RandomForestClassifier
import argparse

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    # -------------------------------
    # Load Wine dataset
    # -------------------------------
    wine = load_wine()
    X = wine.data
    y = wine.target

    # -------------------------------
    # Save dataset to disk
    # -------------------------------
    if not os.path.exists('data'):
        os.makedirs('data')

    with open('data/data.pickle', 'wb') as data_file:
        pickle.dump(X, data_file)

    with open('data/target.pickle', 'wb') as target_file:
        pickle.dump(y, target_file)

    # -------------------------------
    # Configure MLflow
    # -------------------------------
    mlflow.set_tracking_uri("./mlruns")

    dataset_name = "Wine Dataset"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=dataset_name):

        params = {
            "dataset_name": dataset_name,
            "number_of_datapoints": X.shape[0],
            "number_of_dimensions": X.shape[1],
            "num_classes": len(set(y))
        }
        mlflow.log_params(params)

        # -------------------------------
        # Train model
        # -------------------------------
        forest = RandomForestClassifier(random_state=0)
        forest.fit(X, y)

        y_pred = forest.predict(X)
        mlflow.log_metrics({
            'Accuracy': accuracy_score(y, y_pred),
            'F1 Score': f1_score(y, y_pred, average='macro')
        })

        # -------------------------------
        # Save model
        # -------------------------------
        if not os.path.exists('models'):
            os.makedirs('models')

        model_version = f'model_{timestamp}'
        model_filename = f'{model_version}_rf_model.joblib'
        dump(forest, f"models/{model_filename}")
