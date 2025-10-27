# pipeline_scheduler.py
import time
import traceback
from datetime import datetime
from data_ingest import download_dataset, read_raw
from preprocess import load_raw, summarize, build_and_apply_pipeline, load_data, preprocess_data, save_processed_data
from eda import correlation_analysis, univariate_plots, categorical_counts, feature_importance_placeholder
from db import insert_log
import logging
import sys
import os
from prefect import flow, task, get_run_logger

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

@flow()
def run_pipeline_once():
    try:
        insert_log("pipeline", "info", f"Pipeline run started")
        # 1. Ingest
        try:
            download_dataset()
            df_raw = read_raw()
        except Exception as e:
            insert_log("pipeline", "error", f"Ingest failed: {e}")
            return
        # 2. Preprocess
        try:
            summary = summarize(df_raw)
            df_proc = build_and_apply_pipeline(df_raw)
        except Exception as e:
            tb = traceback.format_exc()
            insert_log("pipeline", "error", f"Preprocess failed: {e}\n{tb}")
            return
        # 3. EDA
        try:
            correlation_analysis(df_proc)
            univariate_plots(df_proc)
            categorical_counts(df_raw)
            feature_importance_placeholder(df_proc)
        except Exception as e:
            tb = traceback.format_exc()
            insert_log("pipeline", "error", f"EDA failed: {e}\n{tb}")
            return
        insert_log("pipeline", "success", f"Pipeline run finished successfully")
    except Exception as e:
        tb = traceback.format_exc()
        insert_log("pipeline", "critical", f"Unexpected error: {e}\n{tb}")


def launch_dashboard():
    # Launch Streamlit dashboard as a subprocess
    dashboard_file = os.path.join(os.path.dirname(__file__), "source.py")
    subprocess.Popen(["streamlit", "run", dashboard_file])


if __name__ == "__main__":
    print("Starting pipeline. Running once.")
    run_pipeline_once.serve(name="deploy-4-C")
    print("Launching dashboard...")
    launch_dashboard()