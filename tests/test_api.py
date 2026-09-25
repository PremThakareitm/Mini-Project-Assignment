"""
Integration tests for FastAPI endpoints (/health, /predict, /explain, /metrics).
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_endpoint():
    payload = {
        "investor_name": "Titan Capital",
        "investor_type": "VC",
        "headquarters_city": "Bengaluru",
        "headquarters_state": "Karnataka",
        "founded_year": 2017,
        "investment_stage": "Series A",
        "preferred_sector": "FinTech",
        "portfolio_companies": 45,
        "successful_exits": 8,
        "active_fund": 1,
        "ai_focus": 1,
        "fintech_focus": 1,
        "healthtech_focus": 0,
        "agritech_focus": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "success_fallback"]
    assert data["expected_funding_usd"] > 0
    assert "$" in data["formatted_funding"]


def test_explain_endpoint():
    payload = {
        "investor_name": "Titan Capital",
        "investor_type": "VC",
        "headquarters_city": "Bengaluru",
        "headquarters_state": "Karnataka",
        "founded_year": 2017,
        "investment_stage": "Series A",
        "preferred_sector": "FinTech",
        "portfolio_companies": 45,
        "successful_exits": 8,
        "active_fund": 1,
        "ai_focus": 1,
        "fintech_focus": 1,
        "healthtech_focus": 0,
        "agritech_focus": 0,
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_funding_usd" in data
    assert "positive_factors" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions_served" in data
