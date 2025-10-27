#!/usr/bin/env python3
from flask import Flask, jsonify, g
import sqlite3
import os

DB_PATH = "pipeline_logs.db"
app = Flask(__name__)

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

@app.route("/metrics", methods=["GET"])
def get_all_metrics():
    db = get_db()
    cur = db.execute("SELECT * FROM model_metrics ORDER BY id DESC LIMIT 100")
    rows = cur.fetchall()
    metrics = [dict(r) for r in rows]
    return jsonify(metrics)

@app.route("/metrics/latest", methods=["GET"])
def get_latest():
    db = get_db()
    cur = db.execute("SELECT * FROM model_metrics ORDER BY timestamp DESC LIMIT 10")
    rows = cur.fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/models", methods=["GET"])
def list_models():
    # list model files available
    models_dir = "models"
    files = []
    if os.path.isdir(models_dir):
        files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    return jsonify(files)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)






