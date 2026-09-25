"""
Unit tests for model pipeline fitting and regression metrics.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from src.features.build_features import StartupInvestorFeatureTransformer, build_preprocessor_pipeline
from src.models.train import calculate_regression_metrics


def test_full_pipeline_fit_predict():
    X_train = pd.DataFrame({
        "investor_type": ["VC", "Angel", "Corporate VC"],
        "headquarters_city": ["Bengaluru", "Mumbai", "Delhi NCR"],
        "headquarters_state": ["Karnataka", "Maharashtra", "Delhi"],
        "founded_year": [2010, 2018, 2015],
        "investment_stage": ["Series A", "Seed", "Series B"],
        "preferred_sector": ["FinTech", "DeepTech", "ClimateTech"],
        "portfolio_companies": [40, 10, 25],
        "successful_exits": [8, 1, 4],
        "active_fund": [1, 1, 0],
        "ai_focus": [1, 0, 1],
        "fintech_focus": [1, 1, 0],
        "healthtech_focus": [0, 0, 1],
        "agritech_focus": [0, 0, 0],
    })
    y_train = np.log1p([1_000_000, 300_000, 2_500_000])

    pipeline = Pipeline([
        ("feature_engineer", StartupInvestorFeatureTransformer()),
        ("preprocessor", build_preprocessor_pipeline()),
        ("regressor", Ridge())
    ])

    pipeline.fit(X_train, y_train)

    preds_log = pipeline.predict(X_train)
    assert len(preds_log) == 3
    preds_usd = np.expm1(preds_log)
    assert (preds_usd > 0).all()


def test_regression_metrics_calculation():
    y_true = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])
    y_pred = np.array([1_100_000.0, 1_900_000.0, 3_100_000.0])

    metrics = calculate_regression_metrics(y_true, y_pred)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert "MedianAE" in metrics
    assert "MAPE" in metrics
    assert metrics["MAE"] == pytest.approx(100_000.0)
    assert metrics["R2"] > 0.95
