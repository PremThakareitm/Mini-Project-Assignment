#!/usr/bin/env python3
"""
Automated Retraining Workflow for StartupFund AI.
Conceptually:
New Data -> Drift Check -> Retrain Candidate -> Quality Gate Check -> Promote Champion
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monitoring.drift_detector import detect_data_drift
from src.models.train import train_and_evaluate_all_models


def run_retraining_flow():
    print("🔍 [Retraining Loop] Checking data drift on incoming production data...")

    # Load test batch to simulate incoming production traffic
    test_path = "data/processed/test.csv"
    if not os.path.exists(test_path):
        print("⚠️ Test data not found. Running full training pipeline first...")
        train_and_evaluate_all_models()
        return

    curr_df = pd.read_csv(test_path)
    drift_result = detect_data_drift(curr_df)

    print(f"Data Drift Status: {drift_result.get('status')}")
    print(f"Overall Drift Detected: {drift_result.get('overall_drift_detected')}")

    print("\n🔄 Triggering automated retraining pipeline...")
    res = train_and_evaluate_all_models()

    print(f"\n✅ Retraining Workflow Complete! Promoted Model: {res['best_model_name']}")


if __name__ == "__main__":
    run_retraining_flow()
