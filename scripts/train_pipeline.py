#!/usr/bin/env python3
"""
CLI Script to execute ML model training, tracking, and export.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.train import train_and_evaluate_all_models


def main():
    print("🚀 Starting StartupFund AI Model Training & MLflow Tracking...")
    res = train_and_evaluate_all_models()
    print("\n✅ Training Complete!")
    print(f"Champion Model: {res['best_model_name']}")
    print(f"R² Score: {res['best_r2']:.4f}")
    print(f"Pipeline exported to: {res['model_export_path']}")


if __name__ == "__main__":
    main()
