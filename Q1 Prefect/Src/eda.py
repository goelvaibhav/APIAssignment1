# eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db import insert_log
import numpy as np

PROCESSED_CSV = "data/processed/adult_processed.csv"
EDA_DIR = "data/eda"
os.makedirs(EDA_DIR, exist_ok=True)

def load_processed():
    df = pd.read_csv(PROCESSED_CSV)
    return df

def correlation_analysis(df):
    # numeric columns: choose those that are numeric in processed (original numeric cols are first)
    corr = df.corr()
    corr_file = os.path.join(EDA_DIR, "correlation_matrix.csv")
    corr.to_csv(corr_file)
    # heatmap image
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    heatmap_path = os.path.join(EDA_DIR, "correlation_heatmap.png")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    insert_log("eda", "success", f"Saved correlation matrix and heatmap")
    return corr_file, heatmap_path

def univariate_plots(df):
    # pick a few columns for histograms and boxplots
    hist_paths = []
    for col in df.columns[:6]:
        plt.figure()
        df[col].hist(bins=30)
        plt.title(f"Histogram: {col}")
        p = os.path.join(EDA_DIR, f"hist_{col}.png")
        plt.savefig(p); plt.close()
        hist_paths.append(p)
    # boxplot for hours-per-week-like columns if present
    if "hours-per-week" in df.columns:
        plt.figure()
        df["hours-per-week"].boxplot()
        p = os.path.join(EDA_DIR, "box_hours-per-week.png")
        plt.savefig(p); plt.close()
        hist_paths.append(p)
    insert_log("eda", "success", f"Saved {len(hist_paths)} univariate plots")
    return hist_paths

def categorical_counts(original_df):
    # original_df: we need original raw categories; try to read raw
    try:
        raw = pd.read_csv("data/raw/adult.data.csv", header=None, names=[
            "age","workclass","fnlwgt","education","education-num","marital-status",
            "occupation","relationship","race","sex","capital-gain","capital-loss",
            "hours-per-week","native-country","income"
        ], na_values=["?", " ?"])
    except Exception:
        raw = None
    paths = []
    if raw is not None:
        for col in ["workclass","occupation","education","sex"]:
            plt.figure(figsize=(6,4))
            raw[col].value_counts().nlargest(10).plot(kind='bar')
            plt.title(f"Top counts: {col}")
            p = os.path.join(EDA_DIR, f"count_{col}.png")
            plt.tight_layout()
            plt.savefig(p); plt.close()
            paths.append(p)
        insert_log("eda", "success", f"Saved categorical count plots")
    else:
        insert_log("eda", "warn", "Could not load raw file for categorical counts")
    return paths

def feature_importance_placeholder(df):
    # No model yet; placeholder: use correlation with target as "importance"
    if "income" in df.columns:
        corr_with_target = df.corr()["income"].abs().sort_values(ascending=False)
        path = os.path.join(EDA_DIR, "feature_importance.csv")
        corr_with_target.to_csv(path)
        insert_log("eda", "success", "Saved feature importance (abs corr with target)")
        return path
    else:
        insert_log("eda", "error", "No target column found for feature importance")
        return None

if __name__ == "__main__":
    df = load_processed()
    correlation_analysis(df)
    univariate_plots(df)
    categorical_counts(df)
    feature_importance_placeholder(df)
    print("EDA complete.")