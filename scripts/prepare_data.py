#!/usr/bin/env python3
"""
CLI Script to prepare and clean dataset for DVC pipeline stage.
"""
import os
import sys

# Add root project path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.ingest_validate import load_raw_data, validate_schema
from src.data.clean_data import clean_dataset


def main():
    raw_path = "Indian_Investor_Dataset_2026.csv"
    output_dir = "data/processed"
    output_path = os.path.join(output_dir, "prepared_data.csv")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Ingesting raw dataset from: {raw_path}")
    raw_df = load_raw_data(raw_path)

    # Clean data before schema validation
    cleaned_df = clean_dataset(raw_df)

    is_valid, msg = validate_schema(cleaned_df)
    if not is_valid:
        print(f"❌ Validation Error: {msg}")
        sys.exit(1)

    cleaned_df.to_csv(output_path, index=False)
    print(f"✅ Prepared cleaned dataset saved to: {output_path} ({len(cleaned_df)} rows)")


if __name__ == "__main__":
    main()
