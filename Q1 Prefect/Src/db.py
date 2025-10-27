# db.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, MetaData, Table
from sqlalchemy.orm import sessionmaker
import datetime
import os

DB_PATH = os.environ.get("PIPELINE_DB", "sqlite:///pipeline_logs.db")

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
metadata = MetaData()

logs_table = Table(
    "pipeline_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, default=datetime.datetime.utcnow),
    Column("stage", String(50)),
    Column("status", String(20)),
    Column("message", Text),
)

metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def insert_log(stage: str, status: str, message: str):
    session = Session()
    ins = logs_table.insert().values(timestamp=datetime.datetime.utcnow(),
                                     stage=stage, status=status, message=message)
    session.execute(ins)
    session.commit()
    session.close()

def get_last_logs(limit=50):
    session = Session()
    q = session.execute(logs_table.select().order_by(logs_table.c.timestamp.desc()).limit(limit))
    rows = q.fetchall()
    session.close()
    return rows
