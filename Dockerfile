# Production Dockerfile for StartupFund AI Platform
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Layer Caching Optimization: Copy and install requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dataset and application source code
COPY Indian_Investor_Dataset_2026.csv .
COPY dvc.yaml params.yaml ./
COPY src/ ./src/
COPY api/ ./api/
COPY streamlit/ ./streamlit/
COPY scripts/ ./scripts/
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

# Train model artifact inside image build or ensure pipeline prepared
RUN python3 scripts/train_pipeline.py

EXPOSE 8000 8501

ENTRYPOINT ["/app/entrypoint.sh"]
