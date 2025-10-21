# src/train.py

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data import load_data, build_features, split_data

MODEL_DIR = Path(__file__).resolve().parents[0] / ".." / "model"
MODEL_DIR = MODEL_DIR.resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "walmart_model.pkl"
IMPUTER_PATH = MODEL_DIR / "imputer.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"
INFO_PATH = MODEL_DIR / "model_info.json"


def train(random_state: int = 42):
    df = load_data()
    X, y = build_features(df)

    print(f"Total rows after feature engineering: {len(X)}")

    # simple chronological split (preserves time order)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # impute missing numeric values (fit on train)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # train model
    model = RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=random_state)
    model.fit(X_train_imputed, y_train)

    # evaluate
    preds = model.predict(X_test_imputed)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)

    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE:  {mae:.2f}")

    # persist artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    info = {
        "model_path": str(MODEL_PATH),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "rmse": float(rmse),
        "mae": float(mae)
    }
    with open(INFO_PATH, "w") as fh:
        json.dump(info, fh, indent=2)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved imputer to: {IMPUTER_PATH}")
    print(f"Saved feature list to: {FEATURES_PATH}")
    print(f"Saved model info to: {INFO_PATH}")


if __name__ == "__main__":
    train()
