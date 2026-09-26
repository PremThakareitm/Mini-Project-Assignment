"""
Unit tests for data ingestion, schema validation, and cleaning.
"""

import pytest
import pandas as pd
from src.data.ingest_validate import validate_schema, produce_data_summary
from src.data.clean_data import clean_dataset, CITY_CANONICAL_MAP


def test_validate_schema_valid():
    df = pd.DataFrame({
        "investor_id": ["INV001"],
        "investor_name": ["Test Investor"],
        "investor_type": ["VC"],
        "headquarters_city": ["Mumbai"],
        "headquarters_state": ["Maharashtra"],
        "country": ["India"],
        "founded_year": [2015],
        "investment_stage": ["Series A"],
        "preferred_sector": ["FinTech"],
        "min_investment_inr": [100000],
        "max_investment_inr": [1000000],
        "average_ticket_inr": [500000],
        "portfolio_companies": [20],
        "successful_exits": [5],
        "active_fund": ["Yes"],
        "ai_focus": ["Yes"],
        "fintech_focus": ["Yes"],
        "healthtech_focus": ["No"],
        "agritech_focus": ["No"],
        "website": ["https://example.com"],
        "last_updated": ["2026-07-01"],
    })
    is_valid, msg = validate_schema(df)
    assert is_valid is True
    assert "passed" in msg.lower()


def test_validate_schema_invalid_exits():
    df = pd.DataFrame({
        "investor_id": ["INV001"],
        "investor_name": ["Test Investor"],
        "investor_type": ["VC"],
        "headquarters_city": ["Mumbai"],
        "headquarters_state": ["Maharashtra"],
        "country": ["India"],
        "founded_year": [2015],
        "investment_stage": ["Series A"],
        "preferred_sector": ["FinTech"],
        "min_investment_inr": [100000],
        "max_investment_inr": [1000000],
        "average_ticket_inr": [500000],
        "portfolio_companies": [5],
        "successful_exits": [100],  # Invalid: exits > portfolio
        "active_fund": ["Yes"],
        "ai_focus": ["Yes"],
        "fintech_focus": ["Yes"],
        "healthtech_focus": ["No"],
        "agritech_focus": ["No"],
        "website": ["https://example.com"],
        "last_updated": ["2026-07-01"],
    })
    is_valid, msg = validate_schema(df)
    assert is_valid is False
    assert "exits > portfolio" in msg


def test_clean_dataset_city_canonicalization():
    df = pd.DataFrame({
        "headquarters_city": ["bangalore", "gurgaon", "BOMBay"],
        "active_fund": ["Yes", "No", "1"],
    })
    df_clean = clean_dataset(df)
    assert df_clean["headquarters_city"].iloc[0] == "Bengaluru"
    assert df_clean["headquarters_city"].iloc[1] == "Delhi NCR"
    assert df_clean["headquarters_city"].iloc[2] == "Mumbai"
    assert list(df_clean["active_fund"]) == [1, 0, 1]
