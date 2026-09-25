"""
Pydantic Schemas for FastAPI Input & Output Validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class InvestorPredictionInput(BaseModel):
    investor_name: str = Field(default="Kalaari Capital", description="Name of investor or firm")
    investor_type: str = Field(default="VC", description="Type of investor (Angel, VC, Corporate VC, PE, Family Office)")
    headquarters_city: str = Field(default="Bengaluru", description="Headquarters city")
    headquarters_state: str = Field(default="Karnataka", description="Headquarters state")
    founded_year: int = Field(default=2015, ge=1950, le=2026, description="Year investor was founded")
    investment_stage: str = Field(default="Series A", description="Stage of funding round (Pre-Seed, Seed, Series A, Series B, Series C, Private Equity)")
    preferred_sector: str = Field(default="FinTech", description="Target industry vertical")
    portfolio_companies: int = Field(default=50, ge=0, description="Number of portfolio companies")
    successful_exits: int = Field(default=10, ge=0, description="Number of successful exits")
    active_fund: int = Field(default=1, ge=0, le=1, description="Active fund flag (1 for Yes, 0 for No)")
    ai_focus: int = Field(default=1, ge=0, le=1, description="AI sector focus flag")
    fintech_focus: int = Field(default=1, ge=0, le=1, description="FinTech sector focus flag")
    healthtech_focus: int = Field(default=0, ge=0, le=1, description="HealthTech sector focus flag")
    agritech_focus: int = Field(default=0, ge=0, le=1, description="AgriTech sector focus flag")


class PredictionResponse(BaseModel):
    status: str
    expected_funding_usd: float
    formatted_funding: str
    log_prediction: float
    input_summary: Dict[str, Any]


class ExplanationResponse(BaseModel):
    status: str
    predicted_funding_usd: float
    base_funding_usd: float
    total_delta_usd: float
    positive_factors: List[Dict[str, Any]]
    negative_factors: List[Dict[str, Any]]
    feature_attributions: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    version: str


class MetricsResponse(BaseModel):
    total_predictions_served: int
    average_prediction_usd: float
    last_prediction_timestamp: Optional[str]
