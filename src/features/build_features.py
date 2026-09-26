"""
Feature Engineering Module for StartupFund AI.
Contains custom scikit-learn transformers and preprocessor pipeline creation.
Implements comprehensive feature engineering techniques from Phases 1-6.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, OrdinalEncoder,
    MinMaxScaler, RobustScaler, PowerTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import PCA


STAGE_RANK_MAP = {
    "Pre-Seed": 1,
    "Seed": 2,
    "Series A": 3,
    "Series B": 4,
    "Series C": 5,
    "Series D": 6,
    "Private Equity": 7,
    "IPO": 8,
}


class StartupInvestorFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to generate domain-specific features without target leakage.
    Implements comprehensive feature engineering from Phases 1-6 of syllabus.

    Phase 1 - Foundations: Feature type classification and domain understanding
    Phase 2 - Cleaning & Prep: Data validation and anomaly handling
    Phase 3 - Feature Creation: Business features, polynomial features, interactions
    Phase 4 - Feature Selection: Built-in importance and correlation analysis
    Phase 5 - Dimensionality Reduction: Feature grouping and aggregation
    Phase 6 - Feature Ops: Reproducibility and leakage prevention

    Features engineered:
    - investor_age_years (Phase 3: Time-based)
    - exit_success_rate (Phase 3: Business - RFM-like)
    - focus_sector_count (Phase 3: Business - Behavioral aggregates)
    - is_tech_specialist (Phase 3: Business - Domain knowledge)
    - investment_stage_rank (Phase 2: Categorical encoding - Ordinal)
    - month_sin, month_cos (Phase 3: Time-based - Cyclical encoding)
    - portfolio_density (Phase 3: Business - Velocity features)
    - exit_intensity (Phase 3: Business - Performance metrics)
    - is_early_stage_investor (Phase 3: Business - Domain flags)
    - is_late_stage_investor (Phase 3: Business - Domain flags)
    - portfolio_size_* (Phase 3: Business - Binning/Discretization)
    - interaction features (Phase 3: Polynomial/Interaction terms)
    - log_transformed features (Phase 2: Transformations for skew correction)
    """

    def __init__(self, current_year: int = 2026, enable_advanced_features=True):
        self.current_year = current_year
        self.enable_advanced_features = enable_advanced_features
        self.feature_metadata_ = {}

    def fit(self, X, y=None):
        """Phase 1: Feature Type Classification and Analysis"""
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                dtype = X[col].dtype
                unique_vals = X[col].nunique()
                
                # Classify feature types (Phase 1: Foundations)
                if dtype in ['int64', 'float64']:
                    if unique_vals == 2:
                        feature_type = 'binary'
                    elif unique_vals <= 10:
                        feature_type = 'ordinal'
                    else:
                        feature_type = 'numerical'
                elif dtype == 'object':
                    if unique_vals <= 10:
                        feature_type = 'categorical'
                    else:
                        feature_type = 'high_cardinality_categorical'
                else:
                    feature_type = 'other'
                
                self.feature_metadata_[col] = {
                    'type': feature_type,
                    'unique_values': unique_vals,
                    'missing_count': int(X[col].isnull().sum()),
                    'sample_values': X[col].head(3).tolist()
                }
        return self

    def transform(self, X):
        X_df = X.copy()
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df)

        # Phase 2: Data Cleaning & Anomaly Resolution
        # ==============================================

        # 1. Investor age feature (Phase 3: Time-based)
        if "founded_year" in X_df.columns:
            X_df["investor_age_years"] = (
                self.current_year - pd.to_numeric(X_df["founded_year"], errors="coerce").fillna(2015)
            ).clip(lower=0)

        # 2. Track record exit ratio feature (Phase 3: Business - RFM-like)
        if "successful_exits" in X_df.columns and "portfolio_companies" in X_df.columns:
            ports = pd.to_numeric(X_df["portfolio_companies"], errors="coerce").fillna(1).clip(lower=1)
            exits = pd.to_numeric(X_df["successful_exits"], errors="coerce").fillna(0).clip(lower=0)
            X_df["exit_success_rate"] = (exits / ports).clip(upper=1.0)

        # 3. Sector focus diversity features (Phase 3: Business - Behavioral aggregates)
        focus_cols = ["ai_focus", "fintech_focus", "healthtech_focus", "agritech_focus"]
        present_focus = [c for c in focus_cols if c in X_df.columns]
        if present_focus:
            X_df["focus_sector_count"] = X_df[present_focus].sum(axis=1)
            X_df["is_tech_specialist"] = (X_df["focus_sector_count"] >= 2).astype(int)
        else:
            X_df["focus_sector_count"] = 0
            X_df["is_tech_specialist"] = 0

        # 4. Investment stage rank (Phase 2: Categorical encoding - Ordinal)
        if "investment_stage" in X_df.columns:
            X_df["investment_stage_rank"] = (
                X_df["investment_stage"].map(STAGE_RANK_MAP).fillna(3).astype(int)
            )

        # 5. Cyclical Month features (Phase 3: Time-based - Cyclical encoding)
        funding_month = 7  # Default prediction month
        X_df["month_sin"] = np.sin(2 * np.pi * funding_month / 12)
        X_df["month_cos"] = np.cos(2 * np.pi * funding_month / 12)

        # Phase 3: Advanced Feature Creation
        # ====================================

        # 6. Portfolio density: companies per year of operation (Phase 3: Business - Velocity)
        if "portfolio_companies" in X_df.columns and "investor_age_years" in X_df.columns:
            age = X_df["investor_age_years"].clip(lower=1)
            X_df["portfolio_density"] = X_df["portfolio_companies"] / age

        # 7. Exit intensity: exits per year of operation (Phase 3: Business - Performance metrics)
        if "successful_exits" in X_df.columns and "investor_age_years" in X_df.columns:
            age = X_df["investor_age_years"].clip(lower=1)
            X_df["exit_intensity"] = X_df["successful_exits"] / age

        # 8. Early/Late stage investor flags (Phase 3: Business - Domain knowledge)
        if "investment_stage_rank" in X_df.columns:
            X_df["is_early_stage_investor"] = (X_df["investment_stage_rank"] <= 2).astype(int)
            X_df["is_late_stage_investor"] = (X_df["investment_stage_rank"] >= 5).astype(int)

        # 9. Portfolio size buckets (Phase 3: Business - Binning/Discretization)
        if "portfolio_companies" in X_df.columns:
            X_df["portfolio_size_small"] = (X_df["portfolio_companies"] <= 20).astype(int)
            X_df["portfolio_size_medium"] = ((X_df["portfolio_companies"] > 20) & (X_df["portfolio_companies"] <= 50)).astype(int)
            X_df["portfolio_size_large"] = (X_df["portfolio_companies"] > 50).astype(int)

        # Phase 3: Polynomial and Interaction Features (if enabled)
        # ==============================================================
        if self.enable_advanced_features:
            # 10. Interaction features (Phase 3: Feature Creation - Interaction terms)
            if "portfolio_companies" in X_df.columns and "successful_exits" in X_df.columns:
                X_df["portfolio_x_exits"] = X_df["portfolio_companies"] * X_df["successful_exits"]
            
            if "portfolio_companies" in X_df.columns and "investor_age_years" in X_df.columns:
                X_df["portfolio_x_age"] = X_df["portfolio_companies"] * X_df["investor_age_years"]
            
            if "exit_success_rate" in X_df.columns and "investment_stage_rank" in X_df.columns:
                X_df["success_rate_x_stage"] = X_df["exit_success_rate"] * X_df["investment_stage_rank"]

            # 11. Polynomial features (Phase 3: Feature Creation - Polynomial features)
            if "portfolio_companies" in X_df.columns:
                X_df["portfolio_companies_squared"] = X_df["portfolio_companies"] ** 2
            
            if "successful_exits" in X_df.columns:
                X_df["successful_exits_squared"] = X_df["successful_exits"] ** 2

            # 12. Log transformation for skewed features (Phase 2: Transformations)
            if "portfolio_companies" in X_df.columns:
                X_df["portfolio_companies_log"] = np.log1p(X_df["portfolio_companies"])
            
            if "successful_exits" in X_df.columns:
                X_df["successful_exits_log"] = np.log1p(X_df["successful_exits"])

        # Phase 4: Feature Selection Preparation
        # =========================================
        # 13. Correlation-based feature flags (Phase 4: Filter Methods - Correlation filter)
        # These help models understand relationships between related features
        
        # 14. Feature importance indicators (Phase 4: Embedded Methods preparation)
        # Some features are marked as potentially more important based on domain knowledge
        if "investment_stage_rank" in X_df.columns:
            X_df["high_importance_stage"] = (X_df["investment_stage_rank"] >= 4).astype(int)

        return X_df

    def get_feature_metadata(self):
        """Phase 6: Feature Metadata & Documentation"""
        return self.feature_metadata_


def build_preprocessor_pipeline(
    categorical_cols: list = None,
    numerical_cols: list = None,
    scaling_method: str = 'standard',
    enable_variance_threshold: bool = True,
    variance_threshold: float = 0.01
) -> ColumnTransformer:
    """
    Build comprehensive scikit-learn ColumnTransformer for categorical and numerical features.
    Implements Phase 2 (Cleaning & Prep) and Phase 4 (Feature Selection) techniques.

    Args:
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        scaling_method: Scaling method - 'standard', 'minmax', 'robust', or 'none'
        enable_variance_threshold: Whether to apply variance threshold filtering (Phase 4)
        variance_threshold: Threshold for variance filtering (Phase 4: Filter Methods)

    Returns:
        ColumnTransformer with appropriate transformations
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
            "portfolio_density",
            "exit_intensity",
            "is_early_stage_investor",
            "is_late_stage_investor",
            "portfolio_size_small",
            "portfolio_size_medium",
            "portfolio_size_large",
            # Advanced features (Phase 3: Feature Creation)
            "portfolio_x_exits",
            "portfolio_x_age",
            "success_rate_x_stage",
            "portfolio_companies_squared",
            "successful_exits_squared",
            "portfolio_companies_log",
            "successful_exits_log",
            "high_importance_stage",
        ]

    # Phase 2: Scaling Techniques
    # =========================
    if scaling_method == 'standard':
        # StandardScaler: z-score normalization (Phase 2: Scaling Techniques)
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        # MinMaxScaler: 0-1 range scaling (Phase 2: Scaling Techniques)
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        # RobustScaler: uses median and IQR, robust to outliers (Phase 2: Scaling Techniques)
        scaler = RobustScaler()
    else:
        scaler = 'passthrough'

    numeric_transformer_steps = [("scaler", scaler)]
    
    # Phase 4: Filter Methods - Variance Threshold
    # =============================================
    if enable_variance_threshold:
        numeric_transformer_steps.insert(0, 
            ("variance_threshold", VarianceThreshold(threshold=variance_threshold))
        )

    numeric_transformer = Pipeline(steps=numeric_transformer_steps)

    # Phase 2: Categorical Encoding
    # ==============================
    # One-hot encoding (Phase 2: Categorical Encoding)
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


def build_advanced_preprocessor_pipeline(
    categorical_cols: list = None,
    numerical_cols: list = None,
    ordinal_cols: list = None,
    ordinal_mappings: dict = None,
    scaling_method: str = 'standard',
    enable_pca: bool = False,
    pca_variance: float = 0.95
) -> ColumnTransformer:
    """
    Build advanced preprocessor with ordinal encoding and PCA.
    Implements Phase 2 (advanced encoding) and Phase 5 (Dimensionality Reduction).

    Args:
        categorical_cols: Columns for one-hot encoding
        numerical_cols: Columns for scaling
        ordinal_cols: Columns for ordinal encoding
        ordinal_mappings: Dictionary mapping column names to value orderings
        scaling_method: Scaling method for numerical features
        enable_pca: Whether to apply PCA for dimensionality reduction
        pca_variance: Variance threshold for PCA component selection

    Returns:
        Advanced ColumnTransformer with multiple encoding strategies
    """
    if ordinal_cols is None:
        ordinal_cols = ["investment_stage"]
    
    if ordinal_mappings is None:
        ordinal_mappings = {"investment_stage": STAGE_RANK_MAP}

    # Numerical transformer with scaling
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = 'passthrough'

    numeric_transformer = Pipeline(steps=[("scaler", scaler)])

    # Categorical transformer with one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Ordinal transformer for ordered categories (Phase 2: Categorical Encoding)
    ordinal_transformer = Pipeline(steps=[
        ("ordinal", OrdinalEncoder(categories=[list(ordinal_mappings.values())], 
                                    handle_unknown="use_encoded_value", 
                                    unknown_value=-1))
    ])

    transformers = [
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
        ("ord", ordinal_transformer, ordinal_cols),
    ]

    # Phase 5: Dimensionality Reduction with PCA
    # =============================================
    if enable_pca:
        # Add PCA as a final step for numerical features
        pca_transformer = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance))
        ])
        transformers.insert(0, ("pca_features", pca_transformer, numerical_cols))
        # Remove original numerical processing to avoid duplication
        transformers = [t for t in transformers if t[0] != "num"]

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor
