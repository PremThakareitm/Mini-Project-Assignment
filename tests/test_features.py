"""
Unit tests for custom feature engineering transformations.
"""

import pytest
import pandas as pd
from src.features.build_features import StartupInvestorFeatureTransformer


def test_feature_transformer_calculations():
    transformer = StartupInvestorFeatureTransformer(current_year=2026)

    df_in = pd.DataFrame({
        "founded_year": [2016, 2020],
        "portfolio_companies": [50, 10],
        "successful_exits": [10, 0],
        "ai_focus": [1, 0],
        "fintech_focus": [1, 1],
        "healthtech_focus": [0, 0],
        "agritech_focus": [0, 0],
        "investment_stage": ["Series A", "Pre-Seed"],
    })

    df_out = transformer.transform(df_in)

    assert "investor_age_years" in df_out.columns
    assert "exit_success_rate" in df_out.columns
    assert "focus_sector_count" in df_out.columns
    assert "is_tech_specialist" in df_out.columns
    assert "investment_stage_rank" in df_out.columns

    # Check calculations
    assert df_out["investor_age_years"].iloc[0] == 10
    assert df_out["exit_success_rate"].iloc[0] == pytest.approx(0.20)
    assert df_out["focus_sector_count"].iloc[0] == 2
    assert df_out["is_tech_specialist"].iloc[0] == 1
    assert df_out["investment_stage_rank"].iloc[0] == 3  # Series A rank
    assert df_out["investment_stage_rank"].iloc[1] == 1  # Pre-Seed rank
