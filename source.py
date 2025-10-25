# ...existing code...
# (previous content of this file remains above)
# ...existing code...

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from db import insert_log, get_last_logs

# ensure local module imports work when running the script directly or via streamlit
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# import DB helpers

# Ensure EDA and processed paths exist
PROCESSED_CSV = "data/processed/adult_processed.csv"
EDA_DIR = "data/eda"
os.makedirs(EDA_DIR, exist_ok=True)

# The db functions (insert_log, get_last_logs) are defined earlier in this file
# If they are in a different module in this file, they are available here.

# ----------------------
# EDA logic (appended)
# ----------------------
def load_processed():
    if not os.path.exists(PROCESSED_CSV):
        insert_log("eda", "error", f"Processed CSV not found at {PROCESSED_CSV}")
        raise FileNotFoundError(PROCESSED_CSV)
    df = pd.read_csv(PROCESSED_CSV)
    return df

def correlation_analysis(df):
    try:
        corr = df.corr()
        corr_file = os.path.join(EDA_DIR, "correlation_matrix.csv")
        corr.to_csv(corr_file)
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=False, cmap="coolwarm")
        heatmap_path = os.path.join(EDA_DIR, "correlation_heatmap.png")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()
        insert_log("eda", "success", f"Saved correlation matrix and heatmap")
        return corr_file, heatmap_path
    except Exception as e:
        insert_log("eda", "error", f"correlation_analysis failed: {e}")
        raise

def univariate_plots(df, max_cols=6):
    paths = []
    try:
        for col in df.columns[:max_cols]:
            plt.figure()
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].dropna().hist(bins=30)
                plt.title(f"Histogram: {col}")
            else:
                df[col].value_counts().nlargest(20).plot(kind='bar')
                plt.title(f"Counts: {col}")
            p = os.path.join(EDA_DIR, f"univ_{col}.png")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            paths.append(p)
        # optional boxplot
        if "hours-per-week" in df.columns:
            plt.figure()
            df["hours-per-week"].dropna().plot.box()
            p = os.path.join(EDA_DIR, "box_hours-per-week.png")
            plt.tight_layout()
            plt.savefig(p); plt.close()
            paths.append(p)
        insert_log("eda", "success", f"Saved {len(paths)} univariate plots")
        return paths
    except Exception as e:
        insert_log("eda", "error", f"univariate_plots failed: {e}")
        raise

def categorical_counts():
    paths = []
    try:
        raw_path = "data/raw/adult.data.csv"
        if not os.path.exists(raw_path):
            insert_log("eda", "warn", "Raw CSV for categorical counts not found")
            return paths
        raw = pd.read_csv(raw_path, header=None, names=[
            "age","workclass","fnlwgt","education","education-num","marital-status",
            "occupation","relationship","race","sex","capital-gain","capital-loss",
            "hours-per-week","native-country","income"
        ], na_values=["?", " ?"])
        for col in ["workclass","occupation","education","sex"]:
            if col in raw.columns:
                plt.figure(figsize=(6,4))
                raw[col].value_counts().nlargest(10).plot(kind='bar')
                plt.title(f"Top counts: {col}")
                p = os.path.join(EDA_DIR, f"count_{col}.png")
                plt.tight_layout()
                plt.savefig(p); plt.close()
                paths.append(p)
        insert_log("eda", "success", f"Saved categorical count plots ({len(paths)})")
    except Exception as e:
        insert_log("eda", "error", f"categorical_counts failed: {e}")
    return paths

def feature_importance_placeholder(df):
    try:
        if "income" not in df.columns:
            insert_log("eda", "warn", "No target column 'income' found for feature importance")
            return None
        # use absolute correlation as a quick proxy for importance for numeric columns
        numeric = df.select_dtypes(include=['number'])
        if numeric.shape[1] == 0:
            insert_log("eda", "warn", "No numeric features for importance calculation")
            return None
        corr_with_target = numeric.corrwith(df['income'].map(lambda v: 1 if str(v).strip() in [">50K",">50K."] else 0)).abs()
        corr_with_target = corr_with_target.sort_values(ascending=False)
        path = os.path.join(EDA_DIR, "feature_importance.csv")
        corr_with_target.to_csv(path, header=["importance"])
        plt.figure(figsize=(8,6))
        corr_with_target.head(20).plot(kind='barh')
        plt.gca().invert_yaxis()
        plt.title("Feature importance (abs corr with income)")
        figp = os.path.join(EDA_DIR, "feature_importances.png")
        plt.tight_layout()
        plt.savefig(figp); plt.close()
        insert_log("eda", "success", "Saved feature importance proxy")
        return path
    except Exception as e:
        insert_log("eda", "error", f"feature_importance_placeholder failed: {e}")
        raise

def run_eda_pipeline():
    try:
        df = load_processed()
        correlation_analysis(df)
        univariate_plots(df)
        categorical_counts()
        feature_importance_placeholder(df)
        insert_log("eda", "success", "EDA pipeline completed")
    except Exception as e:
        insert_log("eda", "error", f"EDA pipeline failed: {e}")
        raise

# ----------------------
# Streamlit app logic (appended)
# ----------------------
try:
    import streamlit as st
except Exception:
    st = None

def streamlit_dashboard():
    if st is None:
        print("Streamlit not available. Install streamlit to run dashboard.")
        return

    st.set_page_config(page_title="Data Pipeline Dashboard", layout="wide")
    st.title("Data Pipeline Dashboard — Sub Objective 1")
    st.markdown("Shows recent logs, summary stats, and EDA artifacts (auto-updating).")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        if st.button("Run EDA now"):
            with st.spinner("Running EDA..."):
                try:
                    run_eda_pipeline()
                    st.success("EDA completed")
                except Exception:
                    st.error("EDA failed — check logs")
        if st.button("Refresh logs"):
            pass
        st.write("Recent logs (most recent 100):")
        rows = get_last_logs(100)
        if rows:
            df_logs = pd.DataFrame(rows, columns=["id","timestamp","stage","status","message"])
            st.dataframe(df_logs)
        else:
            st.write("No logs found.")

    # Main area: processed data preview
    st.header("Processed data preview")
    if os.path.exists(PROCESSED_CSV):
        dfp = pd.read_csv(PROCESSED_CSV)
        st.write("Shape:", dfp.shape)
        st.dataframe(dfp.head(10))
    else:
        st.info("Processed CSV not found. Run ingestion & preprocessing first.")

    # EDA artifacts
    st.header("EDA Artifacts")
    if os.path.isdir(EDA_DIR):
        imgs = sorted([os.path.join(EDA_DIR, f) for f in os.listdir(EDA_DIR) if f.lower().endswith(".png")])
        if imgs:
            cols = st.columns(3)
            for i, img in enumerate(imgs):
                with cols[i % 3]:
                    st.image(img, caption=os.path.basename(img), use_column_width=True)
        else:
            st.write("No EDA images yet.")
    else:
        st.write("EDA directory not created yet.")

# If you want to run the dashboard when this file is executed directly (streamlit run uses the file)
if __name__ == "__main__" and st is not None:
    streamlit_dashboard()

# ...existing code...
#```# filepath: c:\Users\vaibhgoel\Desktop\API Assignment\Data Pipeline\.venv\flow.py
# ...existing code...
# (previous content of this file remains above)
# ...existing code...

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure EDA and processed paths exist
PROCESSED_CSV = "data/processed/adult_processed.csv"
EDA_DIR = "data/eda"
os.makedirs(EDA_DIR, exist_ok=True)

# The db functions (insert_log, get_last_logs) are defined earlier in this file
# If they are in a different module in this file, they are available here.

# ----------------------
# EDA logic (appended)
# ----------------------
def load_processed():
    if not os.path.exists(PROCESSED_CSV):
        insert_log("eda", "error", f"Processed CSV not found at {PROCESSED_CSV}")
        raise FileNotFoundError(PROCESSED_CSV)
    df = pd.read_csv(PROCESSED_CSV)
    return df

def correlation_analysis(df):
    try:
        corr = df.corr()
        corr_file = os.path.join(EDA_DIR, "correlation_matrix.csv")
        corr.to_csv(corr_file)
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=False, cmap="coolwarm")
        heatmap_path = os.path.join(EDA_DIR, "correlation_heatmap.png")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()
        insert_log("eda", "success", f"Saved correlation matrix and heatmap")
        return corr_file, heatmap_path
    except Exception as e:
        insert_log("eda", "error", f"correlation_analysis failed: {e}")
        raise

def univariate_plots(df, max_cols=6):
    paths = []
    try:
        for col in df.columns[:max_cols]:
            plt.figure()
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].dropna().hist(bins=30)
                plt.title(f"Histogram: {col}")
            else:
                df[col].value_counts().nlargest(20).plot(kind='bar')
                plt.title(f"Counts: {col}")
            p = os.path.join(EDA_DIR, f"univ_{col}.png")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            paths.append(p)
        # optional boxplot
        if "hours-per-week" in df.columns:
            plt.figure()
            df["hours-per-week"].dropna().plot.box()
            p = os.path.join(EDA_DIR, "box_hours-per-week.png")
            plt.tight_layout()
            plt.savefig(p); plt.close()
            paths.append(p)
        insert_log("eda", "success", f"Saved {len(paths)} univariate plots")
        return paths
    except Exception as e:
        insert_log("eda", "error", f"univariate_plots failed: {e}")
        raise

def categorical_counts():
    paths = []
    try:
        raw_path = "data/raw/adult.data.csv"
        if not os.path.exists(raw_path):
            insert_log("eda", "warn", "Raw CSV for categorical counts not found")
            return paths
        raw = pd.read_csv(raw_path, header=None, names=[
            "age","workclass","fnlwgt","education","education-num","marital-status",
            "occupation","relationship","race","sex","capital-gain","capital-loss",
            "hours-per-week","native-country","income"
        ], na_values=["?", " ?"])
        for col in ["workclass","occupation","education","sex"]:
            if col in raw.columns:
                plt.figure(figsize=(6,4))
                raw[col].value_counts().nlargest(10).plot(kind='bar')
                plt.title(f"Top counts: {col}")
                p = os.path.join(EDA_DIR, f"count_{col}.png")
                plt.tight_layout()
                plt.savefig(p); plt.close()
                paths.append(p)
        insert_log("eda", "success", f"Saved categorical count plots ({len(paths)})")
    except Exception as e:
        insert_log("eda", "error", f"categorical_counts failed: {e}")
    return paths

def feature_importance_placeholder(df):
    try:
        if "income" not in df.columns:
            insert_log("eda", "warn", "No target column 'income' found for feature importance")
            return None
        # use absolute correlation as a quick proxy for importance for numeric columns
        numeric = df.select_dtypes(include=['number'])
        if numeric.shape[1] == 0:
            insert_log("eda", "warn", "No numeric features for importance calculation")
            return None
        corr_with_target = numeric.corrwith(df['income'].map(lambda v: 1 if str(v).strip() in [">50K",">50K."] else 0)).abs()
        corr_with_target = corr_with_target.sort_values(ascending=False)
        path = os.path.join(EDA_DIR, "feature_importance.csv")
        corr_with_target.to_csv(path, header=["importance"])
        plt.figure(figsize=(8,6))
        corr_with_target.head(20).plot(kind='barh')
        plt.gca().invert_yaxis()
        plt.title("Feature importance (abs corr with income)")
        figp = os.path.join(EDA_DIR, "feature_importances.png")
        plt.tight_layout()
        plt.savefig(figp); plt.close()
        insert_log("eda", "success", "Saved feature importance proxy")
        return path
    except Exception as e:
        insert_log("eda", "error", f"feature_importance_placeholder failed: {e}")
        raise

def run_eda_pipeline():
    try:
        df = load_processed()
        correlation_analysis(df)
        univariate_plots(df)
        categorical_counts()
        feature_importance_placeholder(df)
        insert_log("eda", "success", "EDA pipeline completed")
    except Exception as e:
        insert_log("eda", "error", f"EDA pipeline failed: {e}")
        raise

# ----------------------
# Streamlit app logic (appended)
# ----------------------
try:
    import streamlit as st
except Exception:
    st = None

# ...existing code...
def streamlit_dashboard():
    if st is None:
        print("Streamlit not available. Install streamlit to run dashboard.")
        return

    st.set_page_config(page_title="Data Pipeline Dashboard", layout="wide")
    st.title("Data Pipeline Dashboard — Sub Objective 1")
    st.markdown("Shows recent logs, summary stats, and EDA artifacts (auto-updating).")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        # give each button a unique key to avoid StreamlitDuplicateElementId
        if st.button("Run EDA now", key="btn_run_eda"):
            with st.spinner("Running EDA..."):
                try:
                    run_eda_pipeline()
                    st.success("EDA completed")
                except Exception:
                    st.error("EDA failed — check logs")

        if st.button("Run pipeline now", key="btn_run_pipeline"):
            with st.spinner("Running full pipeline..."):
                try:
                    run_pipeline()  # ensure run_pipeline is defined/imported
                    st.success("Pipeline completed")
                except Exception:
                    st.error("Pipeline failed — check logs")

        if st.button("Run pipeline (skip download)", key="btn_run_pipeline_skip"):
            with st.spinner("Running pipeline (skip download)..."):
                try:
                    run_pipeline(skip_download=True)
                    st.success("Pipeline completed (skip download)")
                except Exception:
                    st.error("Pipeline failed — check logs")

        if st.button("Refresh logs", key="btn_refresh_logs"):
            pass

        st.write("Recent logs (most recent 100):")
        rows = get_last_logs(100)
        if rows:
            df_logs = pd.DataFrame(rows, columns=["id","timestamp","stage","status","message"])
            st.dataframe(df_logs)
        else:
            st.write("No logs found.")

    # Main area: processed data preview
    st.header("Processed data preview")
    if os.path.exists(PROCESSED_CSV):
        dfp = pd.read_csv(PROCESSED_CSV)
        st.write("Shape:", dfp.shape)
        st.dataframe(dfp.head(10))
    else:
        st.info("Processed CSV not found. Run ingestion & preprocessing first.")

    # EDA artifacts
    st.header("EDA Artifacts")
    if os.path.isdir(EDA_DIR):
        imgs = sorted([os.path.join(EDA_DIR, f) for f in os.listdir(EDA_DIR) if f.lower().endswith(".png")])
        if imgs:
            cols = st.columns(3)
            for i, img in enumerate(imgs):
                with cols[i % 3]:
                    st.image(img, caption=os.path.basename(img), use_column_width=True)
        else:
            st.write("No EDA images yet.")
    else:
        st.write("EDA directory not created yet.")
# ...existing code...

# If you want to run the dashboard when this file is executed directly (streamlit run uses the file)
if __name__ == "__main__" and st is not None:
    streamlit_dashboard()

# ...existing code...