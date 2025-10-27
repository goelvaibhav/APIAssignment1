"""
preprocess.py
-------------
Comprehensive preprocessing module for Cloud-based Data Science / ML application.

Includes:
- load_raw(): Load raw data.
- summarize(): Display dataset summary and statistics.
- build_and_apply_pipeline(): Automate preprocessing (missing value handling, encoding, scaling).
- load_data(), preprocess_data(), save_processed_data(): Core utility functions.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import os

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(
    filename="pipeline_preprocess.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# Core Functions
# --------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform missing value imputation, encoding, and normalization."""
    logging.info("Starting preprocessing...")
    df = df.copy()

    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna("Unknown", inplace=True)

    # Encode categorical variables
    cat_cols = df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        try:
            df[col] = le.fit_transform(df[col])
            logging.info(f"Encoded column: {col}")
        except Exception as e:
            logging.warning(f"Skipping encoding for {col}: {e}")

    # Normalize numeric variables
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    logging.info("Numeric columns normalized.")

    logging.info("Preprocessing completed successfully.")
    return df


def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed dataset to CSV."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        raise


# --------------------------------------------------
# Extended/Compatible Functionality
# --------------------------------------------------

def load_raw(filepath: str) -> pd.DataFrame:
    """Alias for load_data() – kept for backward compatibility."""
    logging.info("Using load_raw()")
    return load_data(filepath)


def summarize(df: pd.DataFrame):
    """Generate dataset summary, including basic statistics."""
    logging.info("Generating dataset summary...")
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "describe": df.describe(include="all").to_dict()
    }
    logging.info("Summary generated successfully.")
    return summary


def build_and_apply_pipeline(filepath: str, output_path: str) -> pd.DataFrame:
    """
    Complete end-to-end preprocessing pipeline:
    1. Load data
    2. Summarize data
    3. Apply preprocessing
    4. Save processed data
    """
    logging.info("Running build_and_apply_pipeline()")
    df = load_raw(filepath)
    _ = summarize(df)
    processed_df = preprocess_data(df)
    save_processed_data(processed_df, output_path)
    logging.info("Pipeline completed successfully.")
    return processed_df


# --------------------------------------------------
# Main Execution (Manual Run)
# --------------------------------------------------
if __name__ == "__main__":
    INPUT_PATH = "data/raw/adult.data.csv"
    OUTPUT_PATH = "data/processed/adult_processed.csv"

    df = load_raw(INPUT_PATH)
    processed_df = build_and_apply_pipeline(INPUT_PATH, OUTPUT_PATH)
    print("✅ Preprocessing complete. Processed file saved successfully.")
