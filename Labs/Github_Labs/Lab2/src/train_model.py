import argparse
import datetime
import os
from joblib import dump
import mlflow
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":

    # -----------------------------
    # 1️⃣ Parse Arguments
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp from GitHub Actions")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory to store models, data, and metrics")
    args = parser.parse_args()

    timestamp = args.timestamp
    base_dir = args.base_dir

    print(f"Timestamp received → {timestamp}")
    print(f"Base directory → {base_dir}")

    # Create folders inside base_dir
    model_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # -----------------------------
    # 2️⃣ Load Wine dataset
    # -----------------------------
    data = load_wine()
    X, y = data.data, data.target
    print(f"Dataset loaded → X: {X.shape}, y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(
        f"Train/Test split → X_train: {X_train.shape}, X_test: {X_test.shape}")

    # -----------------------------
    # 3️⃣ Train Model
    # -----------------------------
    forest = RandomForestClassifier(random_state=42)
    forest.fit(X_train, y_train)
    print("Model trained successfully ✅")

    # -----------------------------
    # 4️⃣ Evaluate Model
    # -----------------------------
    y_pred = forest.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")  # multiclass-safe

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # -----------------------------
    # 5️⃣ MLflow Logging
    # -----------------------------
    mlflow.set_tracking_uri(os.path.join(base_dir, "mlruns"))

    experiment_name = f"wine_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="RandomForest_Wine"):
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("algorithm", "RandomForestClassifier")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

    # -----------------------------
    # 6️⃣ Save Model Artifact
    # -----------------------------
    model_path = os.path.join(model_dir, f"model_{timestamp}_rf_model.joblib")
    dump(forest, model_path)
    print(f"Model saved → {model_path}")

    # -----------------------------
    # 7️⃣ Save Test Split
    # -----------------------------
    test_split_path = os.path.join(data_dir, "test_split.joblib")
    dump((X_test, y_test), test_split_path)
    print(f"Test split saved → {test_split_path}")

    print("🎉 Training pipeline completed successfully!")
