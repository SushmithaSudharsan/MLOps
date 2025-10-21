# Walmart Sales Forecasting API (FastAPI + MLOps Lab)

This project demonstrates how to **train, package, and serve a Machine Learning model as a REST API** using **FastAPI** and **Uvicorn**.  
The model predicts **weekly Walmart store sales** using lag, rolling, and date-based features derived from historical data.

---

## 📘 Overview

### What you’ll learn
- Train a **RandomForestRegressor** on the Walmart dataset.  
- Engineer **lag**, **rolling**, and **date-based** features for time-series forecasting.  
- Save and load serialized model artifacts (`.pkl` files).  
- Expose the trained model as a REST API using **FastAPI**.  
- Interact with your model using a browser or API calls.

---

## 🧩 Project Structure

MLOps Labs/
└── Labs/
└── API_Labs/
└── FastAPI_Labs/
├── assets/
├── model/
│ ├── walmart_model.pkl
│ ├── imputer.pkl
│ ├── feature_columns.pkl
│ └── model_info.json
├── src/
│ ├── init.py
│ ├── data.py
│ ├── train.py
│ ├── predict.py
│ └── main.py
├── Walmart.csv
├── README.md
└── requirements.txt

yaml
Copy code

---

## 🧠 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python 3.9+ |
| API Framework | FastAPI, Uvicorn |
| ML / Data | Pandas, NumPy, scikit-learn |
| Model Type | RandomForestRegressor |
| Serialization | joblib |
| Environment | Virtualenv / venv |

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
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Verify dataset
Make sure the file Walmart.csv is present inside the main FastAPI_Labs/ folder.

🧮 Training the Model
Run the training script to:

Load and clean the dataset

Engineer features (lags, rolling means, date components)

Train a RandomForest model

Save all model artifacts to model/

bash
Copy code
cd src
python train.py
Expected output:

yaml
Copy code
Loading data...
Training RandomForestRegressor...
Test RMSE: 2320.45
Test MAE: 1809.23
Saved model to: ../model/walmart_model.pkl
Artifacts created:

pgsql
Copy code
model/
├── walmart_model.pkl
├── imputer.pkl
├── feature_columns.pkl
└── model_info.json
🚀 Running the FastAPI App
Start the API server:

bash
Copy code
uvicorn main:app --reload
Output:

arduino
Copy code
INFO:     Uvicorn running on http://127.0.0.1:8000
Open in browser:
👉 http://127.0.0.1:8000/docs

🧪 Testing the API
🩺 Health Check
Endpoint: GET /

Response:

json
Copy code
{
  "status": "healthy",
  "message": "Walmart sales forecasting API"
}
📈 Model Info
Endpoint: GET /model-info

Returns model metrics (RMSE, MAE, training size, etc.).

🔮 Predict Weekly Sales
Endpoint: POST /predict

Example Request Body:

json
Copy code
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
Response:

json
Copy code
{
  "predicted_weekly_sales": 13025.47
}
🧩 API Endpoints Summary
Method	Endpoint	Description
GET	/	Health check
GET	/model-info	Returns model metadata
POST	/predict	Predict weekly Walmart sales
POST	/feature_engineer	(Optional) Generate lag & rolling features

🧰 Utility Modules
File	Description
data.py	Loads and preprocesses the Walmart dataset. Builds lag, rolling, and date features.
train.py	Trains a RandomForest model and saves all artifacts.
predict.py	Loads trained model and returns predictions for API inputs.
main.py	Defines FastAPI routes and integrates prediction logic.

🧠 How It Works
csharp
Copy code
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
     Saved Artifacts: model/, imputer/, features/
                    │
                    ▼
          [main.py → FastAPI server]
                    │
                    ▼
          Client calls /predict → JSON prediction
🛑 Stopping the Server
Press CTRL + C in the terminal running FastAPI.

💡 Future Improvements
Deploy API to Render, Hugging Face Spaces, or AWS Lambda.

Automate retraining and version control (MLOps pipeline).

Add Streamlit dashboard for visual forecasting.

Containerize using Docker for production deployment.

👩‍💻 Author
Sushmitha Sudharsan
Graduate Student – Data Analytics Engineering

Focused on Data Science, Analytics, and MLOps applications.