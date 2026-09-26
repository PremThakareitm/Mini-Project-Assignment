"""
Data Ingestion & Validation Module for StartupFund AI.
Provides functions to load raw dataset, perform structural validation,
and return summary statistics.
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd


EXPECTED_COLUMNS = [
    "investor_id",
    "investor_name",
    "investor_type",
    "headquarters_city",
    "headquarters_state",
    "country",
    "founded_year",
    "investment_stage",
    "preferred_sector",
    "min_investment_inr",
    "max_investment_inr",
    "average_ticket_inr",
    "portfolio_companies",
    "successful_exits",
    "active_fund",
    "ai_focus",
    "fintech_focus",
    "healthtech_focus",
    "agritech_focus",
    "website",
    "last_updated",
]


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw dataset CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df


def validate_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate dataset schema and integrity rules.
    Returns (is_valid, error_message).
    """
    # Check expected columns presence
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    # Check non-empty dataset
    if df.empty:
        return False, "Dataset is empty."

    # Validate target column numeric range
    if "average_ticket_inr" in df.columns:
        if (df["average_ticket_inr"] < 0).any():
            return False, "Found negative values in average_ticket_inr target."

    # Validate logical bounds (e.g. successful_exits <= portfolio_companies)
    if "successful_exits" in df.columns and "portfolio_companies" in df.columns:
        invalid_exits = (df["successful_exits"] > df["portfolio_companies"]).sum()
        if invalid_exits > 0:
            return False, f"Found {invalid_exits} records where successful_exits > portfolio_companies."

    return True, "Dataset schema and constraints validation passed."


def produce_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Produce comprehensive summary report of the dataset."""
    summary = {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "cardinality": {col: int(df[col].nunique()) for col in df.columns},
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    if "average_ticket_inr" in df.columns:
        target_series = df["average_ticket_inr"]
        summary["target_distribution"] = {
            "min": float(target_series.min()),
            "max": float(target_series.max()),
            "mean": float(target_series.mean()),
            "std": float(target_series.std()),
            "median": float(target_series.median()),
            "q25": float(target_series.quantile(0.25)),
            "q75": float(target_series.quantile(0.75)),
            "skew": float(target_series.skew()),
        }

    return summary
