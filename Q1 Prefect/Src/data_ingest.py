# data_ingest.py
import os
import requests
import pandas as pd
from db import insert_log

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)
RAW_CSV = os.path.join(RAW_DIR, "adult.data.csv")
COLUMN_FILE = os.path.join(RAW_DIR, "columns.txt")

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"

DEFAULT_COLS = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","income"
]

def download_dataset():
    try:
        r = requests.get(DATA_URL, timeout=15)
        r.raise_for_status()
        text = r.text.strip()
        # Save CSV (UCI has comma separated lines)
        with open(RAW_CSV, "w", encoding="utf-8") as f:
            f.write(text)
        insert_log("ingest", "success", f"Downloaded dataset to {RAW_CSV}")
        return RAW_CSV
    except Exception as e:
        insert_log("ingest", "error", f"Download failed: {e}")
        raise

def read_raw():
    # Read into pandas using default column names
    df = pd.read_csv(RAW_CSV, header=None, names=DEFAULT_COLS, na_values=["?", " ?"])
    insert_log("ingest", "success", f"Read raw csv with shape {df.shape}")
    return df

if __name__ == "__main__":
    download_dataset()
    df = read_raw()
    print("Ingested rows:", len(df))
