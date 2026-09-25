"""
Data Cleaning Module for StartupFund AI.
Provides reproducible cleaning transformations:
- City canonicalization mapping
- Categorical text normalization
- Binary flag conversion (Yes/No -> 1/0)
- Date formatting
- Data anomaly resolution (capping successful_exits <= portfolio_companies)
"""

import pandas as pd
import numpy as np


# Canonical city mapping for Indian startup hubs
CITY_CANONICAL_MAP = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "delhi": "Delhi NCR",
    "new delhi": "Delhi NCR",
    "gurgaon": "Delhi NCR",
    "gurugram": "Delhi NCR",
    "noida": "Delhi NCR",
    "hyderabad": "Hyderabad",
    "chennai": "Chennai",
    "pune": "Pune",
    "kolkata": "Kolkata",
    "ahmedabad": "Ahmedabad",
}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply reproducible cleaning steps to raw DataFrame."""
    df_clean = df.copy()

    # 1. Clean string columns (trim whitespace)
    str_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # 2. Canonicalize city names
    if "headquarters_city" in df_clean.columns:
        df_clean["headquarters_city"] = (
            df_clean["headquarters_city"]
            .str.lower()
            .map(lambda c: CITY_CANONICAL_MAP.get(c, c.title()))
        )

    # 3. Convert Binary flags ('Yes'/'No' -> 1/0)
    binary_cols = ["active_fund", "ai_focus", "fintech_focus", "healthtech_focus", "agritech_focus"]
    for col in binary_cols:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.upper()
                .map({"YES": 1, "NO": 0, "1": 1, "0": 0})
                .fillna(0)
                .astype(int)
            )

    # 4. Standardize investment_stage text format
    if "investment_stage" in df_clean.columns:
        df_clean["investment_stage"] = df_clean["investment_stage"].str.title()

    # 5. Standardize preferred_sector text format
    if "preferred_sector" in df_clean.columns:
        df_clean["preferred_sector"] = df_clean["preferred_sector"].str.title()

    # 6. Parse and standardize dates
    if "last_updated" in df_clean.columns:
        df_clean["funding_date"] = pd.to_datetime(df_clean["last_updated"], errors="coerce").fillna(pd.Timestamp("2026-01-01"))

    # 7. Data Anomaly Resolution: Cap successful_exits <= portfolio_companies
    if "successful_exits" in df_clean.columns and "portfolio_companies" in df_clean.columns:
        ports = pd.to_numeric(df_clean["portfolio_companies"], errors="coerce").fillna(1).astype(int)
        exits = pd.to_numeric(df_clean["successful_exits"], errors="coerce").fillna(0).astype(int)
        # Cap exits at portfolio_companies
        df_clean["successful_exits"] = np.minimum(exits, ports)
        df_clean["portfolio_companies"] = ports

    return df_clean
