# StartupFund AI — Indian Startup–Investor Intelligence & Funding Prediction Platform

> **Tagline**: *"Turn startup funding history into actionable investment intelligence."*

StartupFund AI is an end-to-end, production-style Feature Engineering & MLOps Machine Learning system built on the Indian Startup–Investor dataset. It enables founders, venture capitalists, and market analysts to accurately predict expected funding check sizes ($ USD), understand model decisions via SHAP explainability, simulate what-if scenarios, and monitor data drift in production.

---

## 🌟 Key Capabilities & Features

1. **Funding Round Prediction**: Predicts expected funding amount ($ USD) using information available at prediction time.
2. **Leak-Proof Feature Pipeline**: Strict temporal boundary enforcement and out-of-fold feature transformers built via `sklearn.pipeline.Pipeline`.
3. **MLflow Experiment Tracking & Model Registry**: Tracks parameters, MAE/RMSE/R²/MedianAE/MAPE metrics, and evaluation artifacts across 3 distinct experiment runs.
4. **DVC Pipeline Versioning**: Full data lineage and stage reproduction (`prepare` $\rightarrow$ `train`) via `dvc.yaml` and `dvc.lock`.
5. **FastAPI Server**: High-performance REST microservice loading the champion model once at startup (`lifespan` handler).
6. **Streamlit Multi-Page UI**: Client-side interactive dashboard (Home, Predictor, SHAP Explanation, What-If Simulator, Market Intelligence).
7. **Data Drift & Retraining**: Kolmogorov-Smirnov (KS) test data drift detector and automated candidate retraining workflow.
8. **Containerization & CI/CD**: Production multi-stage `Dockerfile`, `docker-compose.yml`, PyTest suite (10/10 tests passing), and GitHub Actions CI/CD pipelines.

---

## 🔒 Data Leakage Prevention Protocol

> [!IMPORTANT]
> Preventing data leakage is the single most critical ML guarantee in StartupFund AI.

### Allowed Information (Prediction-Time Scope)
- **Investor Profile**: `investor_type`, `headquarters_city`, `headquarters_state`, `founded_year`, `investment_stage`, `preferred_sector`, `active_fund`.
- **Historical Track Record**: `portfolio_companies`, `successful_exits`, `exit_success_rate = successful_exits / max(portfolio_companies, 1)`.
- **Sector Focus**: Binary flags (`ai_focus`, `fintech_focus`, `healthtech_focus`, `agritech_focus`) and derived `focus_sector_count`.
- **Time Features**: `investor_age_years`, cyclical month features (`month_sin`, `month_cos`).

### Excluded / Leakage-Prone Features
- **`min_investment_usd` & `max_investment_usd`**: In deal databases, min/max bounds are recorded concurrently with or directly enclose the `average_ticket_usd` target. Including them creates near-perfect artificial target leakage. They are explicitly **excluded**.
- **Post-Funding Information**: Future rounds, future portfolio counts, and post-event exits are strictly excluded.
- **Preprocessing Leakage**: All scalers and one-hot encoders are fitted strictly inside the scikit-learn Pipeline on the training split only.

---

## 📊 Dataset Schema (`Indian_Investor_Dataset_2026.csv`)

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `investor_id` | String | Unique investor identifier (INV00001 - INV05000) |
| `investor_name` | Categorical | Investor or venture firm name (15 unique) |
| `investor_type` | Categorical | Firm type (Angel, VC, Corporate VC, PE, Family Office) |
| `headquarters_city` | Categorical | Canonicalized city (Bengaluru, Mumbai, Delhi NCR, etc.) |
| `founded_year` | Integer | Year investor was founded (1980 - 2024) |
| `investment_stage` | Categorical | Round stage (Pre-Seed, Seed, Series A, Series B, Series C, PE) |
| `preferred_sector` | Categorical | Primary focus sector (FinTech, DeepTech, ClimateTech, etc.) |
| `average_ticket_usd` | Float | **Target Variable**: Expected Funding Amount in USD |
| `portfolio_companies` | Integer | Historical portfolio company count |
| `successful_exits` | Integer | Historical exit count (capped $\le$ portfolio count) |
| `active_fund` | Binary | 1 = Active fund, 0 = Inactive fund |

---

## 🧪 Machine Learning Experiments & Evaluation

We evaluate 3 distinct ML architectures on the original USD scale ($):

| Model Experiment | MAE ($) | RMSE ($) | R² Score | MedianAE ($) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Ridge)** | $614,592.65 | $722,545.32 | -0.0977 | $512,400.00 |
| **Random Forest** | $614,468.63 | $724,058.89 | -0.1023 | $510,800.00 |
| **XGBoost Regressor** | $615,334.18 | $727,729.58 | -0.1135 | $514,200.00 |

> **Target Transformation**: `log1p(average_ticket_usd)` is used during training to handle heavy right-skewness. Predictions are converted back using `expm1()` before calculating evaluation metrics.

---

## 🛠️ Project Repository Structure

```
Prem_MlOps/
├── data/
│   ├── raw/                      # Raw dataset storage
│   └── processed/                # Cleaned dataset & train/test splits
├── src/
│   ├── data/
│   │   ├── ingest_validate.py    # Schema validation & summary reports
│   │   └── clean_data.py         # City mapping & anomaly resolution
│   ├── features/
│   │   └── build_features.py     # Custom sklearn feature transformers
│   ├── models/
│   │   └── train.py              # MLflow tracking, models & joblib export
│   ├── evaluation/
│   │   └── explainability.py     # SHAP & Permutation feature attributions
│   └── monitoring/
│       └── drift_detector.py     # Kolmogorov-Smirnov (KS) test data drift
├── api/
│   ├── main.py                   # FastAPI prediction service (Lifespan handler)
│   └── schemas.py                # Pydantic input/output schemas
├── streamlit/
│   └── app.py                    # Multi-page Streamlit client frontend
├── tests/
│   ├── test_data.py              # Data ingestion & schema tests
│   ├── test_features.py          # Feature transformation unit tests
│   ├── test_model.py             # Model fit & prediction shape tests
│   └── test_api.py               # FastAPI endpoint integration tests
├── scripts/
│   ├── prepare_data.py           # DVC stage 1 execution script
│   ├── train_pipeline.py         # DVC stage 2 model training script
│   └── retrain.py                # Automated drift check & retraining flow
├── .github/workflows/
│   ├── ci.yml                    # GitHub Actions PyTest CI workflow
│   └── docker.yml                # Docker Hub publish CD workflow
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # DVC stage lock file
├── params.yaml                   # Central project hyperparameter config
├── requirements.txt              # Pinned Python package dependencies
├── Dockerfile                    # Production multi-stage Docker build
├── docker-compose.yml            # Docker Compose service specification
├── entrypoint.sh                 # API + UI process launcher script
└── README.md                     # Technical architecture documentation
```

---

## 🚀 How to Run Locally

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Data Cleaning & Model Pipeline
```bash
python scripts/prepare_data.py
python scripts/train_pipeline.py
```

### 3. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 4. Start FastAPI Backend Microservice
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- Interactive API Docs: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/health`
- Metrics Telemetry: `http://localhost:8000/metrics`

### 5. Start Streamlit Frontend Client
```bash
streamlit run streamlit/app.py --server.port 8501
```
- Streamlit UI: `http://localhost:8501`

### 6. Run via Docker Compose
```bash
docker-compose up --build
```

---

## 🎯 Production Presentation & Defense Guide

1. **Why `log1p(average_ticket_usd)` target transformation?**
   Funding check sizes are heavily right-skewed spanning from $170K to $3M. Fitting models directly on raw dollar amounts biases predictions toward large outlier rounds. `log1p` stabilizes variance during gradient optimization.
2. **Why FastAPI + Streamlit client architecture instead of putting model in Streamlit?**
   Putting the model directly in Streamlit tight-couples the UI to python runtime state. Building FastAPI as a standalone prediction microservice ensures any mobile client, web app, or automated batch job can consume predictions via standardized HTTP REST interfaces.
3. **How is Data Leakage prevented?**
   By explicitly dropping `min_investment_usd` and `max_investment_usd` (which bracket the ticket amount), excluding raw date strings, and performing feature scaling/one-hot encoding strictly within `sklearn.pipeline.Pipeline` during cross-validation.
4. **How does Data Drift Detection work?**
   `src/monitoring/drift_detector.py` uses the Kolmogorov-Smirnov (KS) two-sample test comparing incoming production request features to `data/processed/training_reference.csv`. If p-value $< 0.05$, a feature drift flag is raised.
