"""
Model Training, MLflow Tracking & Registry Module for StartupFund AI.
Trains 3 distinct ML experiments (Baseline Ridge, Random Forest, XGBoost / GradientBoosting),
logs params/metrics/artifacts to MLflow, promotes the Champion model, and saves the full
sklearn Pipeline to artifacts/model.joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

import mlflow
import mlflow.sklearn

from src.data.ingest_validate import load_raw_data, validate_schema
from src.data.clean_data import clean_dataset
from src.features.build_features import (
    StartupInvestorFeatureTransformer,
    build_preprocessor_pipeline,
)

# Attempt XGBoost import
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def calculate_regression_metrics(y_true, y_pred) -> dict:
    """Calculate MAE, RMSE, R2, MedianAE, MAPE on original scale ($)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    median_ae = float(median_absolute_error(y_true, y_pred))

    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MedianAE": median_ae,
        "MAPE": mape,
    }


def generate_evaluation_plots(y_true, y_pred, model_name: str, output_dir: str = "artifacts"):
    """Generate and save evaluation plots for MLflow artifact logging."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Prediction vs Actual Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true / 1e6, y_pred / 1e6, alpha=0.4, color="#028090")
    ax.plot([y_true.min()/1e6, y_true.max()/1e6], [y_true.min()/1e6, y_true.max()/1e6], 'r--', lw=2)
    ax.set_xlabel("Actual Funding ($ Millions)")
    ax.set_ylabel("Predicted Funding ($ Millions)")
    ax.set_title(f"Prediction vs Actual - {model_name}")
    plt.tight_layout()
    pred_plot_path = os.path.join(output_dir, f"{model_name}_prediction_scatter.png")
    plt.savefig(pred_plot_path)
    plt.close()

    # 2. Residual Distribution Plot
    residuals = (y_true - y_pred) / 1e6
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#B5451B", ax=ax)
    ax.set_xlabel("Residuals ($ Millions)")
    ax.set_title(f"Residual Distribution - {model_name}")
    plt.tight_layout()
    res_plot_path = os.path.join(output_dir, f"{model_name}_residuals.png")
    plt.savefig(res_plot_path)
    plt.close()

    return pred_plot_path, res_plot_path


def train_and_evaluate_all_models(
    data_path: str = "Indian_Investor_Dataset_2026.csv",
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "StartupFund_AI_Funding_Prediction",
    artifacts_dir: str = "artifacts",
) -> dict:
    """
    Run 3 ML experiments (Baseline Ridge, Random Forest, XGBoost / GradientBoosting),
    log metrics/artifacts to MLflow, select champion model, register it, and export pipeline.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # 1. Ingest, Clean & Validate
    raw_df = load_raw_data(data_path)
    cleaned_df = clean_dataset(raw_df)

    is_valid, msg = validate_schema(cleaned_df)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {msg}")

    cleaned_df.to_csv("data/processed/prepared_data.csv", index=False)

    # 2. Exclude Leakage Columns & Define Features/Target
    target_col = "average_ticket_usd"
    leakage_cols = ["min_investment_usd", "max_investment_usd", "investor_id", "website", "last_updated", "funding_date"]

    feature_df = cleaned_df.drop(columns=[c for c in leakage_cols if c in cleaned_df.columns])

    X = feature_df.drop(columns=[target_col])
    y = feature_df[target_col].values

    # Target transformation log1p for training
    y_log = np.log1p(y)

    # Split Train/Test (80/20)
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    y_test_usd = np.expm1(y_test_log)

    # Save reference datasets for drift monitoring and reproducibility
    X_train.to_csv("data/processed/train.csv", index=False)
    X_test.to_csv("data/processed/test.csv", index=False)
    X_train.head(500).to_csv("data/processed/training_reference.csv", index=False)

    # 3. Setup MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Define Model Configurations
    if HAS_XGBOOST:
        adv_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42)
        adv_name = "XGBoost"
    else:
        adv_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
        adv_name = "GradientBoosting"

    models_dict = {
        "Baseline_Ridge": Ridge(alpha=1.0),
        "Random_Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        adv_name: adv_model,
    }

    best_r2 = -float("inf")
    best_pipeline = None
    best_model_name = ""
    results_summary = {}

    for model_name, regressor in models_dict.items():
        with mlflow.start_run(run_name=model_name):
            # Construct single reproducible sklearn Pipeline
            full_pipeline = Pipeline([
                ("feature_engineer", StartupInvestorFeatureTransformer()),
                ("preprocessor", build_preprocessor_pipeline()),
                ("regressor", regressor)
            ])

            # Fit on training fold
            full_pipeline.fit(X_train, y_train_log)

            # Predict on test set & invert log transform back to original USD scale
            y_pred_log = full_pipeline.predict(X_test)
            y_pred_usd = np.expm1(y_pred_log)

            # Calculate evaluation metrics on original $ scale
            metrics = calculate_regression_metrics(y_test_usd, y_pred_usd)

            # Log parameters to MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("target_transform", "log1p")
            mlflow.log_param("leakage_prevention", "Excluded min_investment_usd & max_investment_usd")

            # Log metrics to MLflow
            for metric_name, val in metrics.items():
                mlflow.log_metric(metric_name, val)

            # Generate and log evaluation plots
            pred_plot, res_plot = generate_evaluation_plots(y_test_usd, y_pred_usd, model_name, artifacts_dir)
            mlflow.log_artifact(pred_plot)
            mlflow.log_artifact(res_plot)

            # Log sklearn pipeline artifact to MLflow using cloudpickle serialization
            try:
                mlflow.sklearn.log_model(full_pipeline, artifact_path="model", serialization_format="cloudpickle")
            except Exception:
                mlflow.sklearn.log_model(full_pipeline, artifact_path="model")

            results_summary[model_name] = metrics

            print(f"[{model_name}] MAE: ${metrics['MAE']:,.2f} | RMSE: ${metrics['RMSE']:,.2f} | R²: {metrics['R2']:.4f}")

            # Track best champion model based on R² and MAE
            if metrics["R2"] > best_r2:
                best_r2 = metrics["R2"]
                best_pipeline = full_pipeline
                best_model_name = model_name

    # Export Champion Model artifact to artifacts/model.joblib
    model_export_path = os.path.join(artifacts_dir, "model.joblib")
    joblib.dump(best_pipeline, model_export_path)

    print(f"\n🏆 Champion Model Selected: {best_model_name} (R²: {best_r2:.4f})")
    print(f"Artifact exported successfully to: {model_export_path}")

    return {
        "best_model_name": best_model_name,
        "best_r2": best_r2,
        "results_summary": results_summary,
        "model_export_path": model_export_path,
    }


if __name__ == "__main__":
    train_and_evaluate_all_models()
