# src/data.py

import pandas as pd
import numpy as np
from pathlib import Path
from dateutil import parser as date_parser
from typing import Optional, Dict, Any


DATA_FILENAME = "Walmart.csv"


def _data_path() -> Path:
    return Path(__file__).resolve().parents[0] / ".." / DATA_FILENAME


def load_data() -> pd.DataFrame:
    """
    Load and basic-clean the Walmart CSV.
    Returns a DataFrame with columns at least:
      Store, Date, Weekly_Sales, (optional) IsHoliday, Temperature, Fuel_Price, CPI, Unemployment
    """
    path = _data_path().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    # normalize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    if "Date" not in df.columns:
        raise ValueError("The dataset must have a 'Date' column.")

    # keep original string to allow robust parsing
    df["_date_str"] = df["Date"].astype(str)

    # attempt fast parse (dayfirst) then fallback to dateutil for any failures
    df["Date"] = pd.to_datetime(df["_date_str"], dayfirst=True, infer_datetime_format=True, errors="coerce")

    mask = df["Date"].isna()
    if mask.any():
        def _parse_or_nat(s: str):
            try:
                return date_parser.parse(s, dayfirst=True)
            except Exception:
                try:
                    return date_parser.parse(s)
                except Exception:
                    return pd.NaT
        df.loc[mask, "Date"] = df.loc[mask, "_date_str"].apply(_parse_or_nat)

    if df["Date"].isna().any():
        n_fail = int(df["Date"].isna().sum())
        sample = df[df["Date"].isna()].head(10)["_date_str"].to_list()
        raise ValueError(f"{n_fail} date(s) could not be parsed. Sample values: {sample}")

    df = df.drop(columns=["_date_str"])

    # If dept-level data exists, aggregate to Store-Date (sum Weekly_Sales)
    if "Dept" in df.columns:
        agg_map = {"Weekly_Sales": "sum"}
        # include known exogenous fields if present
        for c in ("IsHoliday", "Temperature", "Fuel_Price", "CPI", "Unemployment"):
            if c in df.columns:
                # for IsHoliday take max (if any dept is holiday mark week as holiday),
                # for numeric exog take mean
                agg_map[c] = "max" if c == "IsHoliday" else "mean"
        df = df.groupby(["Store", "Date"], as_index=False).agg(agg_map)

    # ensure Store and Weekly_Sales exist
    if "Store" not in df.columns or "Weekly_Sales" not in df.columns:
        raise ValueError("Required columns missing after load/aggregate: need Store and Weekly_Sales")

    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # fill numeric NA with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


def build_features(df: pd.DataFrame):
    """
    Create lag/rolling/date features for each store.
    Returns:
        X (DataFrame): feature matrix
        y (Series): target Weekly_Sales
    """
    df = df.copy().sort_values(["Store", "Date"])

    for lag in (1, 2, 3, 4):
        df[f"lag_{lag}"] = df.groupby("Store")["Weekly_Sales"].shift(lag)

    for w in (4, 8):
        df[f"roll_mean_{w}"] = (
            df.groupby("Store")["Weekly_Sales"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    # pandas 1.1+: isocalendar returns DataFrame; handle backwards compatibility
    try:
        df["week"] = df["Date"].dt.isocalendar().week.astype(int)
    except Exception:
        df["week"] = df["Date"].dt.week.astype(int)  # fallback (may be deprecated)

    df = df.dropna(subset=["lag_1"]).reset_index(drop=True)

    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_4",
        "roll_mean_4", "roll_mean_8",
        "IsHoliday", "Temperature", "Fuel_Price",
        "CPI", "Unemployment", "year", "month", "week", "Store"
    ]

    # ensure all expected feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    X = df[feature_cols]
    y = df["Weekly_Sales"]

    return X, y


def split_data(X, y, train_frac: float = 0.8):
    """
    Chronological split into train/test.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")

    n = len(X)
    cutoff = int(n * train_frac)
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def prepare_single_feature_row(
    store: int,
    date: str,
    lag_1: Optional[float] = None,
    lag_2: Optional[float] = None,
    lag_3: Optional[float] = None,
    lag_4: Optional[float] = None,
    roll_mean_4: Optional[float] = None,
    roll_mean_8: Optional[float] = None,
    is_holiday: Optional[int] = 0,
    temperature: Optional[float] = None,
    fuel_price: Optional[float] = None,
    cpi: Optional[float] = None,
    unemployment: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the trained model's feature schema.
    Missing values are set to np.nan (imputer should handle them).
    """
    try:
        dt = pd.to_datetime(date)
    except Exception:
        # try dateutil parse as last resort
        try:
            dt = date_parser.parse(str(date))
        except Exception:
            raise ValueError("date must be parseable (YYYY-MM-DD)")

    row: Dict[str, Any] = {
        "lag_1": float(lag_1) if lag_1 is not None else np.nan,
        "lag_2": float(lag_2) if lag_2 is not None else np.nan,
        "lag_3": float(lag_3) if lag_3 is not None else np.nan,
        "lag_4": float(lag_4) if lag_4 is not None else np.nan,
        "roll_mean_4": float(roll_mean_4) if roll_mean_4 is not None else np.nan,
        "roll_mean_8": float(roll_mean_8) if roll_mean_8 is not None else np.nan,
        "IsHoliday": int(is_holiday) if is_holiday is not None else 0,
        "Temperature": float(temperature) if temperature is not None else np.nan,
        "Fuel_Price": float(fuel_price) if fuel_price is not None else np.nan,
        "CPI": float(cpi) if cpi is not None else np.nan,
        "Unemployment": float(unemployment) if unemployment is not None else np.nan,
        "year": int(dt.year),
        "month": int(dt.month),
        "week": int(dt.isocalendar().week),
        "Store": int(store),
    }

    return pd.DataFrame([row])


if __name__ == "__main__":
    df = load_data()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Data prepared successfully.")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")