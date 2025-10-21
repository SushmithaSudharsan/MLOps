# src/main.py
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import datetime

from predict import predict_from_dict  # prediction helper (returns float)
from data import prepare_single_feature_row  # builds the feature-row expected by model

app = FastAPI(
    title="Walmart Weekly Sales Forecast API",
    description="Predict weekly sales per store. Provide pre-engineered lag/rolling features (or compute them elsewhere) and the API will return predicted weekly sales.",
    version="1.0.0",
)

class SalesInput(BaseModel):
    """
    Minimum required: store, date (YYYY-MM-DD), lag_1 (previous week's sales).
    Other lag/rolling/exogenous features are optional but recommended.
    """
    store: int = Field(..., example=1, description="Store ID")
    date: str = Field(..., example="2012-11-02", description="Target week date in YYYY-MM-DD")
    lag_1: float = Field(..., example=12345.67, description="Previous week's total sales for the store")
    lag_2: Optional[float] = Field(None, example=11000.0)
    lag_3: Optional[float] = Field(None, example=10000.0)
    lag_4: Optional[float] = Field(None, example=9500.0)
    roll_mean_4: Optional[float] = Field(None, example=11250.0)
    roll_mean_8: Optional[float] = Field(None, example=10800.0)
    is_holiday: Optional[int] = Field(0, example=0, description="1 if the target week is a holiday week, else 0")
    temperature: Optional[float] = Field(None, example=65.2)
    fuel_price: Optional[float] = Field(None, example=3.7)
    cpi: Optional[float] = Field(None, example=211.0)
    unemployment: Optional[float] = Field(None, example=8.1)

class SalesResponse(BaseModel):
    predicted_weekly_sales: float

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy", "message": "Walmart sales forecasting API"}

@app.post("/predict", response_model=SalesResponse, status_code=status.HTTP_200_OK)
async def predict_sales(input: SalesInput):
    """
    Predict weekly sales for a single store-date.
    The API expects at least lag_1; other features are optional.
    """
    try:
        # Validate / parse date
        try:
            date_obj = datetime.date.fromisoformat(input.date)
        except Exception:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail="date must be in YYYY-MM-DD format")

        # Build feature row in the exact shape expected by model (using helper)
        # prepare_single_feature_row returns a pd.DataFrame with one row
        row_df = prepare_single_feature_row(
            store=input.store,
            date=date_obj,
            lag_1=input.lag_1,
            lag_2=input.lag_2,
            lag_3=input.lag_3,
            lag_4=input.lag_4,
            roll_mean_4=input.roll_mean_4,
            roll_mean_8=input.roll_mean_8,
            is_holiday=input.is_holiday if input.is_holiday is not None else 0,
            temperature=input.temperature,
            fuel_price=input.fuel_price,
            cpi=input.cpi,
            unemployment=input.unemployment,
        )

        # Convert the single-row DataFrame to dict and call prediction helper
        feature_dict = row_df.iloc[0].to_dict()
        pred = predict_from_dict(feature_dict)  # returns float

        return SalesResponse(predicted_weekly_sales=float(pred))

    except FileNotFoundError as e:
        # Model or artifacts missing
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except HTTPException:
        # re-raise HTTP exceptions we intentionally created
        raise
    except Exception as e:
        # Unexpected error
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Prediction error: {e}")
