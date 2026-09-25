"""
Feature Engineering Module for StartupFund AI.
Contains custom scikit-learn transformers and preprocessor pipeline creation.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline


STAGE_RANK_MAP = {
    "Pre-Seed": 1,
    "Seed": 2,
    "Series A": 3,
    "Series B": 4,
    "Series C": 5,
    "Private Equity": 6,
}


class StartupInvestorFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to generate domain-specific features without target leakage.
    Features engineered:
    - investor_age_years
    - exit_success_rate
    - focus_sector_count
    - is_tech_specialist
    - investment_stage_rank
    - month_sin, month_cos
    """

    def __init__(self, current_year: int = 2026):
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy()
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df)

        # 1. Investor age feature
        if "founded_year" in X_df.columns:
            X_df["investor_age_years"] = (
                self.current_year - pd.to_numeric(X_df["founded_year"], errors="coerce").fillna(2015)
            ).clip(lower=0)

        # 2. Track record exit ratio feature
        if "successful_exits" in X_df.columns and "portfolio_companies" in X_df.columns:
            ports = pd.to_numeric(X_df["portfolio_companies"], errors="coerce").fillna(1).clip(lower=1)
            exits = pd.to_numeric(X_df["successful_exits"], errors="coerce").fillna(0).clip(lower=0)
            X_df["exit_success_rate"] = (exits / ports).clip(upper=1.0)

        # 3. Sector focus diversity features
        focus_cols = ["ai_focus", "fintech_focus", "healthtech_focus", "agritech_focus"]
        present_focus = [c for c in focus_cols if c in X_df.columns]
        if present_focus:
            X_df["focus_sector_count"] = X_df[present_focus].sum(axis=1)
            X_df["is_tech_specialist"] = (X_df["focus_sector_count"] >= 2).astype(int)
        else:
            X_df["focus_sector_count"] = 0
            X_df["is_tech_specialist"] = 0

        # 4. Investment stage rank
        if "investment_stage" in X_df.columns:
            X_df["investment_stage_rank"] = (
                X_df["investment_stage"].map(STAGE_RANK_MAP).fillna(3).astype(int)
            )

        # 5. Cyclical Month features (assuming current execution date or month=7)
        funding_month = 7  # Default prediction month
        X_df["month_sin"] = np.sin(2 * np.pi * funding_month / 12)
        X_df["month_cos"] = np.cos(2 * np.pi * funding_month / 12)

        return X_df


def build_preprocessor_pipeline(
    categorical_cols: list = None,
    numerical_cols: list = None
) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for categorical and numerical features.
    """
    if categorical_cols is None:
        categorical_cols = [
            "investor_type",
            "headquarters_city",
            "headquarters_state",
            "investment_stage",
            "preferred_sector",
        ]

    if numerical_cols is None:
        numerical_cols = [
            "founded_year",
            "portfolio_companies",
            "successful_exits",
            "investor_age_years",
            "exit_success_rate",
            "focus_sector_count",
            "is_tech_specialist",
            "investment_stage_rank",
            "month_sin",
            "month_cos",
        ]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    return preprocessor
