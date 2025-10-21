# 🏪 Walmart Sales Forecasting API (FastAPI + MLOps Lab)

This project demonstrates how to **train, deploy, and serve a Machine Learning model as an API** using **FastAPI** and **Uvicorn**.
The model predicts **weekly Walmart store sales** using lag, rolling, and date-based features derived from historical data.

---

## 📘 Overview

### Key Objectives

* Train a **RandomForestRegressor** on the Walmart dataset.
* Engineer **lag**, **rolling**, and **date-based** features for time-series forecasting.
* Save and reload serialized model artifacts (`.pkl` files).
* Serve the trained model as an API using **FastAPI**.
* Test predictions through Swagger UI or API calls.

---

## 🧩 Project Structure

MLOps Labs/
└── Labs/
  └── API_Labs/
    └── FastAPI_Labs/
     ├── assets/
     ├── model/
     │   ├── walmart_model.pkl
     │   ├── imputer.pkl
     │   ├── feature_columns.pkl
     │   └── model_info.json
     ├── src/
     │   ├── **init**.py
     │   ├── data.py
     │   ├── train.py
     │   ├── predict.py
     │   └── main.py
     ├── Walmart.csv
     ├── README.md
     └── requirements.txt

---

## 🧠 Tech Stack

| Category             | Tools / Libraries           |
| -------------------- | --------------------------- |
| Programming Language | Python 3.9+                 |
| API Framework        | FastAPI, Uvicorn            |
| ML / Data            | Pandas, NumPy, scikit-learn |
| Model                | RandomForestRegressor       |
| Serialization        | joblib                      |
| Environment          | Virtualenv / venv           |

---

## ⚙️ Setup Instructions

### 1️⃣ Create and activate a virtual environment

```bash
cd "MLOps Labs/Labs/API_Labs/FastAPI_Labs"
python -m venv fastapi_env

# Windows
fastapi_env\Scripts\activate
# macOS / Linux
source fastapi_env/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Verify dataset

Ensure that **Walmart.csv** exists in the main `FastAPI_Labs` directory.

---

## 🧮 Train the Model

Run the training script to:

* Load and preprocess data
* Engineer features
* Train a RandomForest model
* Save all artifacts to the `model/` folder

```bash
cd src
python train.py
```

Expected output:

```
Loading data...
Training RandomForestRegressor...
Test RMSE: 2320.45
Test MAE: 1809.23
Saved model to: ../model/walmart_model.pkl
```

Artifacts created:

```
model/
├── walmart_model.pkl
├── imputer.pkl
├── feature_columns.pkl
└── model_info.json
```

---

## 🚀 Run the FastAPI Server

Start the API:

```bash
uvicorn main:app --reload
```

Output:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Open Swagger UI:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Test the API

### 🩺 Health Check

**Endpoint:** `GET /`

**Response:**

```json
{
  "status": "healthy",
  "message": "Walmart sales forecasting API"
}
```

---

### 📈 Model Info

**Endpoint:** `GET /model-info`
Returns model metrics (RMSE, MAE, train/test size, etc.).

---

### 🔮 Predict Weekly Sales

**Endpoint:** `POST /predict`

**Example Request Body:**

```json
{
  "store": 1,
  "date": "2012-11-02",
  "lag_1": 13500.5,
  "lag_2": 12890.3,
  "lag_3": 12400.6,
  "lag_4": 11980.0,
  "roll_mean_4": 12700.8,
  "roll_mean_8": 12560.2,
  "is_holiday": 0,
  "temperature": 65.0,
  "fuel_price": 3.6,
  "cpi": 211.2,
  "unemployment": 8.1
}
```

**Response:**

```json
{
  "predicted_weekly_sales": 13025.47
}
```

---

## 🧩 API Endpoints Summary

| Method | Endpoint            | Description                                |
| ------ | ------------------- | ------------------------------------------ |
| GET    | `/`                 | Health check                               |
| GET    | `/model-info`       | Model metadata                             |
| POST   | `/predict`          | Predict weekly Walmart sales               |
| POST   | `/feature_engineer` | (Optional) Generate lag & rolling features |

---

## 🧰 Module Descriptions

| File           | Description                                                                     |
| -------------- | ------------------------------------------------------------------------------- |
| **data.py**    | Loads and cleans Walmart dataset. Builds lag, rolling, and date-based features. |
| **train.py**   | Trains RandomForest model, evaluates it, and saves artifacts.                   |
| **predict.py** | Loads saved model and returns predictions for API input.                        |
| **main.py**    | Implements FastAPI endpoints for health check, prediction, and model info.      |

---

## 🧠 How It Works

```
         ┌───────────────┐
         │  Walmart.csv  │
         └──────┬────────┘
                │
                ▼
     [data.py → feature engineering]
                │
                ▼
    [train.py → RandomForestRegressor]
                │
                ▼
```

Saved Artifacts → model/, imputer/, features/
│
▼
[main.py → FastAPI server]
│
▼
Client → /predict → JSON prediction

---

## 🛑 Stop the Server

Press **CTRL + C** in the terminal to stop the FastAPI server.

---

## 💡 Future Improvements

* Deploy API to **Render**, **Hugging Face Spaces**, or **AWS Lambda**
* Automate retraining and CI/CD integration
* Add a Streamlit dashboard for visualization
* Containerize using Docker for production deployment
