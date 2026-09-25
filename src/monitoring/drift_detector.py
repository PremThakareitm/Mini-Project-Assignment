"""
Data Drift Detection Module for StartupFund AI.
Performs statistical comparison between incoming production request data and
training reference data using the Kolmogorov-Smirnov (KS) test.
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def detect_data_drift(
    current_df: pd.DataFrame,
    reference_path: str = "data/processed/training_reference.csv",
    p_value_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Run KS test comparing numerical distributions of current batch vs reference data.
    """
    if not os.path.exists(reference_path):
        return {
            "status": "warning",
            "message": f"Reference dataset not found at {reference_path}. Train model first.",
            "overall_drift_detected": False,
            "feature_reports": {},
        }

    ref_df = pd.read_csv(reference_path)
    
    num_cols = ["founded_year", "portfolio_companies", "successful_exits"]
    present_num_cols = [c for c in num_cols if c in ref_df.columns and c in current_df.columns]

    feature_reports = {}
    any_drift = False

    for col in present_num_cols:
        ref_vals = pd.to_numeric(ref_df[col], errors="coerce").dropna()
        curr_vals = pd.to_numeric(current_df[col], errors="coerce").dropna()

        if len(curr_vals) < 3 or len(ref_vals) < 3:
            feature_reports[col] = {
                "statistic": 0.0,
                "p_value": 1.0,
                "drift_detected": False,
                "note": "Insufficient sample size for KS test",
            }
            continue

        ks_stat, p_val = ks_2samp(ref_vals, curr_vals)
        is_drifted = bool(p_val < p_value_threshold)
        if is_drifted:
            any_drift = True

        feature_reports[col] = {
            "statistic": float(round(ks_stat, 4)),
            "p_value": float(round(p_val, 4)),
            "drift_detected": is_drifted,
        }

    return {
        "status": "success",
        "overall_drift_detected": any_drift,
        "p_value_threshold": p_value_threshold,
        "feature_reports": feature_reports,
    }
