from flask import Flask, jsonify
import sqlite3
import os

app = Flask(__name__)
DB_PATH = "pipeline_logs.db"

def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("pipeline_logs.db not found. Run ML pipeline first.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Cloud ML Application API",
        "available_endpoints": ["/metrics", "/latest", "/health"]
    })

@app.route('/metrics')
def get_metrics():
    conn = get_db_connection()
    data = conn.execute("SELECT model_name, accuracy, precision, recall, f1, timestamp FROM model_metrics").fetchall()
    conn.close()
    metrics_list = [dict(row) for row in data]
    return jsonify({
        "total_records": len(metrics_list),
        "metrics": metrics_list
    })

@app.route('/latest')
def get_latest_model():
    conn = get_db_connection()
    latest = conn.execute("""
        SELECT model_name, accuracy, precision, recall, f1, timestamp
        FROM model_metrics ORDER BY id DESC LIMIT 1
    """).fetchone()
    conn.close()
    if latest:
        return jsonify(dict(latest))
    else:
        return jsonify({"error": "No model found"}), 404

@app.route('/health')
def health_check():
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1")
        conn.close()
        return jsonify({"status": "healthy"})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
