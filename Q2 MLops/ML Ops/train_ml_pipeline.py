import os
import sys
import sqlite3
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ML_Pipeline_Experiment_3")

# Paths
PROCESSED_PATH = "data/processed/adult_processed.csv"
RAW_PATH = "data/raw/adult.data.csv"
MODELS_DIR = "models"
DB_PATH = "pipeline_logs.db"

RANDOM_STATE = 42

def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def load_or_preprocess():
    """
    Load processed CSV if exists, otherwise load raw CSV and do simple preprocessing.
    Expect adult dataset where last column is the target income ('<=50K' or '>50K')
    """
    if os.path.exists(PROCESSED_PATH):
        print(f"Loading processed dataset from {PROCESSED_PATH}")
        df = pd.read_csv(PROCESSED_PATH)
        return df

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Neither processed data found at {PROCESSED_PATH} nor raw data at {RAW_PATH}."
        )

    print(f"No processed CSV found. Loading raw dataset from {RAW_PATH} and preprocessing...")
    # Adult dataset usually has no header; adjust if header exists.
    col_names = [
        "age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race","sex",
        "capital-gain","capital-loss","hours-per-week","native-country","income"
    ]

    df = pd.read_csv(RAW_PATH, header=None, names=col_names, na_values=[" ?", "?"])
    # Strip whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop rows with missing values (simple approach)
    before = df.shape[0]
    df.dropna(inplace=True)
    after = df.shape[0]
    print(f"Dropped {before - after} rows with missing values.")

    # Convert target to standard labels if needed
    df['income'] = df['income'].replace({'<=50K.':'<=50K','>50K.':'>50K'})
    # Save processed
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed dataset saved to {PROCESSED_PATH}")
    return df

def prepare_features(df):
    # Identify feature columns and target
    target_col = "income"
    if target_col not in df.columns:
        raise KeyError("Expected 'income' column as target")

    X = df.drop(columns=[target_col])
    y = df[target_col].apply(lambda v: 1 if str(v).strip() in (">50K", ">50K.") else 0)

    # Separate numeric and categorical columns (simple heuristic)
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Build preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    return X, y, preprocessor

def init_db(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        train_size INTEGER,
        test_size INTEGER,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1 REAL
    )
    """)
    conn.commit()

def log_metrics(conn, model_name, train_size, test_size, metrics):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO model_metrics
        (model_name, timestamp, train_size, test_size, accuracy, precision, recall, f1)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_name,
        datetime.utcnow().isoformat(),
        train_size,
        test_size,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"]
    ))
    conn.commit()

def evaluate_and_log(model_pipeline, X_test, y_test, model_name, conn, train_size, test_size):
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    print(f"=== {model_name} evaluation ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 score:  {metrics['f1']:.4f}")
    # Log into sqlite
    log_metrics(conn, model_name, train_size, test_size, metrics)
    return metrics

def main():
    ensure_dirs()
    df = load_or_preprocess()
    X, y, preprocessor = prepare_features(df)

    # Train/test split 70%/30% stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(f"Train size: {train_size}, Test size: {test_size}")

    # Connect to sqlite DB and ensure table
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # Define models
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "svm": SVC(probability=True, random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
        # "xgboost": XGBClassifier(random_state=RANDOM_STATE),
        # "lightgbm": LGBMClassifier(random_state=RANDOM_STATE)
    }

    # For each model build a pipeline: preprocessor + model
        # For each model build a pipeline: preprocessor + model
    for name, estimator in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator)
            ])

            print(f"\nTraining {name} ...")
            pipeline.fit(X_train, y_train)
            # Save artifact
            model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
            joblib.dump(pipeline, model_path)
            print(f"Saved model to {model_path}")

            # Evaluate and log metrics
            metrics = evaluate_and_log(pipeline, X_test, y_test, name, conn, train_size, test_size)

            # Log parameters, metrics, and model to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_param("train_size", train_size)
            mlflow.log_param("test_size", test_size)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, name)
        
        # Evaluate and log metrics
        evaluate_and_log(pipeline, X_test, y_test, name, conn, train_size, test_size)
        
        
    conn.close()
    print("\nAll models trained, evaluated, and metrics logged to", DB_PATH)



if __name__ == "__main__":
    main()

