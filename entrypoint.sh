#!/bin/bash
set -e

echo "🚀 Starting StartupFund AI FastAPI Backend Server on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI backend to initialize
echo "Waiting for FastAPI backend server..."
sleep 3

echo "🚀 Starting StartupFund AI Streamlit Frontend Client on port 8501..."
streamlit run streamlit/app.py --server.port 8501 --server.address 0.0.0.0
