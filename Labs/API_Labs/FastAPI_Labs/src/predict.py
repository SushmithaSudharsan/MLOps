# src/predict.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# artifact paths
BASE = Path(__file__).resolve().parents[0] / ".."
MODEL_PATH = (BASE / "model" / "walmart_model.pkl").resolve()
IMPUTER_PATH = (BASE / "model" / "imputer.pkl").resolve()
FEATURES_PATH = (BASE / "model" / "feature_columns.pkl").resolve()

# lazy-loaded globals
_MODEL = None
_IMPUTER = None
_FEATURES = None


def _load_artifacts():
    global _MODEL, _IMPUTER, _FEATURES
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
        _MODEL = joblib.load(MODEL_PATH)
    if _IMPUTER is None:
        if not IMPUTER_PATH.exists():
            raise FileNotFoundError(f"Imputer not found at: {IMPUTER_PATH}")
        _IMPUTER = joblib.load(IMPUTER_PATH)
    if _FEATURES is None:
        if not FEATURES_PATH.exists():
            raise FileNotFoundError(
                f"Feature list not found at: {FEATURES_PATH}")
        _FEATURES = joblib.load(FEATURES_PATH)
    return _MODEL, _IMPUTER, _FEATURES


def _prepare_array_from_df(df: pd.DataFrame, features: List[str], imputer) -> np.ndarray:
    # Ensure all expected columns present
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
    X = df[features].copy()
    # imputer expects numeric numpy array
    X_imputed = imputer.transform(X)
    return X_imputed


def predict_from_dict(d: Dict[str, Any]) -> float:
    """
    Predict a single row passed as a dictionary (feature name -> value).
    Returns a float.
    """
    model, imputer, features = _load_artifacts()
    df = pd.DataFrame([d])
    X = _prepare_array_from_df(df, features, imputer)
    pred = model.predict(X)
    return float(pred[0])


def predict_from_row(df_row: pd.DataFrame) -> float:
    """
    Predict using a single-row DataFrame (pandas), returns float.
    """
    model, imputer, features = _load_artifacts()
    X = _prepare_array_from_df(df_row, features, imputer)
    pred = model.predict(X)
    return float(pred[0])


def predict_batch(list_of_dicts: List[Dict[str, Any]]) -> List[float]:
    """
    Predict a batch of rows given as list of dictionaries.
    Returns a list of floats (predictions in same order).
    """
    model, imputer, features = _load_artifacts()
    df = pd.DataFrame(list_of_dicts)
    X = _prepare_array_from_df(df, features, imputer)
    preds = model.predict(X)
    return [float(x) for x in preds]


# back-compat alias (if any older code calls predict_data)
def predict_data(X):
    """
    Backwards-compatible helper. If X is a list/ndarray (already aligned features),
    convert to numpy and predict. Prefer predict_from_dict/predict_batch for API inputs.
    """
    model, imputer, features = _load_artifacts()
    arr = np.array(X)
    # if input is 1D, reshape
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    preds = model.predict(arr)
    return preds
