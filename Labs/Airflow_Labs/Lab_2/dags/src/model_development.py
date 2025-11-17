# File: src/model_development.py
import os
import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

WORKING_DIR = "./working_data"
MODEL_DIR = "./model"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data() -> str:
    """
    Load sklearn wine dataset and save as pickle.
    Returns path to saved file.
    """
    data = load_wine(as_frame=True)
    df = data.frame
    out_path = os.path.join(WORKING_DIR, "raw.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    return out_path


def data_preprocessing(file_path: str) -> str:
    """
    Load dataframe, split into X/y, scale, and save as pickle.
    Returns path to saved file.
    """
    with open(file_path, "rb") as f:
        df = pickle.load(f)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    out_path = os.path.join(WORKING_DIR, "preprocessed.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((X_train_scaled, X_test_scaled,
                    y_train.values, y_test.values), f)

    return out_path


def separate_data_outputs(file_path: str) -> str:
    """
    Passthrough for DAG composition.
    """
    return file_path


def build_model(file_path: str, filename: str) -> str:
    """
    Train Logistic Regression on Wine dataset and save model.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path


def load_model(file_path: str, filename: str) -> int:
    """
    Load saved model and test set, print accuracy, return first prediction as int.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    score = model.score(X_test, y_test)
    print(f"Model score on test data: {score:.4f}")

    pred = model.predict(X_test)
    return int(pred[0])
