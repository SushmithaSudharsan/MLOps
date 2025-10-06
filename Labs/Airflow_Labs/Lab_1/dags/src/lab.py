import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)  # bytes
    # JSON-safe string
    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model on the preprocessed data and saves it in working_data/model.
    Returns the SSE list (JSON-serializable).
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {"init": "random", "n_init": 10,
                     "max_iter": 300, "random_state": 42}
    sse = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    # Save the last-fitted model in working_data/model
    output_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), "working_data/model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)

    return sse


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report k.
    Returns the first prediction (as a plain int) for test.csv.
    """
    # Load the model from working_data/model
    output_path = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), "working_data/model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # Elbow info
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters: {kl.elbow}")

    # Predict on raw test data
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../data/test.csv"))
    pred = loaded_model.predict(df)[0]

    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred


def save_cluster_assignments(ti, model_filename="model.sav", output_filename="cluster_assignments.csv"):
    """
    Predict cluster assignments using the saved KMeans model and save to a CSV file.
    Pulls preprocessed data from XCom.
    """
    import pandas as pd

    # Pull preprocessed data from XCom
    data_b64 = ti.xcom_pull(task_ids='data_preprocessing_task')
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    # Load the trained model
    model_path = os.path.join(os.path.dirname(os.path.dirname(
        __file__)), "working_data/model", model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    kmeans = pickle.load(open(model_path, "rb"))

    # Predict cluster labels
    cluster_labels = kmeans.predict(df)

    # Save cluster assignments as CSV in working_data
    output_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), "working_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    df_assignments = pd.DataFrame({"Cluster": cluster_labels})
    df_assignments.to_csv(output_path, index=False)

    print(f"Cluster assignments saved to {output_path}")
    return output_path
