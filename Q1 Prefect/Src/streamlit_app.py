# streamlit_app.py
import streamlit as st
from db import get_last_logs
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Data Pipeline Dashboard", layout="wide")

st.title("Data Pipeline Dashboard — Sub Objective 1")
st.markdown("Shows recent logs, summary stats, and EDA artifacts (auto-updating).")

# Logs
st.header("Recent Logs")
rows = get_last_logs(100)
if rows:
    df_logs = pd.DataFrame(rows, columns=["id","timestamp","stage","status","message"])
    st.dataframe(df_logs)
else:
    st.write("No logs yet.")

# Show summary stats (if available)
st.header("Latest Summary & Processed Data Info")
processed_path = "data/processed/adult_processed.csv"
if os.path.exists(processed_path):
    df_proc = pd.read_csv(processed_path)
    st.subheader("Processed data preview")
    st.dataframe(df_proc.head(10))
    st.write("Shape:", df_proc.shape)
else:
    st.write("Processed data not found yet. Wait for a pipeline run.")

# Display EDA images
st.header("EDA Artifacts")
eda_dir = "data/eda"
if os.path.isdir(eda_dir):
    files = sorted([os.path.join(eda_dir,f) for f in os.listdir(eda_dir) if f.endswith(".png")])
    if files:
        cols = st.columns(3)
        for i, f in enumerate(files):
            idx = i % 3
            with cols[idx]:
                st.image(f, caption=os.path.basename(f), use_column_width=True)
    else:
        st.write("No EDA images yet.")
else:
    st.write("EDA directory not created yet.")
