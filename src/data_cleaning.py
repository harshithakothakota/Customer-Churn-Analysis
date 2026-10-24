
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def load_and_clean_data(file_path):
    """Load, validate, and clean the customer churn dataset."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path.resolve()}"
        )

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Required columns missing: {sorted(missing_columns)}"
        )

    # Clean whitespace in text columns.
    text_columns = df.select_dtypes(include=["object"]).columns
    for column in text_columns:
        df[column] = df[column].str.strip()

    # Convert numeric columns safely.
    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[column] = pd.to_numeric(
            df[column].replace("", np.nan),
            errors="coerce",
        )

    # Remove rows without essential identifiers or target labels.
    df = df.dropna(subset=["customerID", "Churn"])
    df = df.drop_duplicates(subset=["customerID"], keep="first")

    # Remove rows with missing essential numeric values.
    df = df.dropna(
        subset=["tenure", "MonthlyCharges", "TotalCharges"]
    )

    # Keep only valid churn labels.
    df = df[df["Churn"].isin(["Yes", "No"])].copy()

    if df.empty:
        raise ValueError("No valid customer records remain.")

    # Feature engineering.
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, np.inf],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-60 months",
            "60+ months",
        ],
    )

    return df.reset_index(drop=True)


def generate_quality_report(df):
    """Calculate data-quality and churn summary metrics."""

    return {
        "rows_after_cleaning": len(df),
        "columns_after_cleaning": len(df.columns),
        "duplicate_customer_ids": int(
            df["customerID"].duplicated().sum()
        ),
        "missing_values": int(df.isna().sum().sum()),
        "churn_rate_percent": round(
            df["ChurnFlag"].mean() * 100, 2
        ),
    }
