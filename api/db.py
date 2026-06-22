"""Layer 7 storage. SQLite holds the prediction/alert history queried by the
API on load; InfluxDB holds the raw sensor time-series. InfluxDB writes are
best-effort so the system still runs if Influx is not up.
"""
from __future__ import annotations
import sqlite3
import datetime as dt
from config import (SQLITE_PATH, INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG,
                    INFLUX_BUCKET)

_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    node    INTEGER NOT NULL,
    model   TEXT NOT NULL,
    prob    REAL NOT NULL,
    tier    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts);
"""


def init_db():
    con = sqlite3.connect(SQLITE_PATH)
    con.executescript(_DDL)
    con.commit()
    return con


def save_predictions(rows):
    """rows: list of (ts, node, model, prob, tier)."""
    con = sqlite3.connect(SQLITE_PATH)
    con.executemany(
        "INSERT INTO predictions(ts,node,model,prob,tier) VALUES (?,?,?,?,?)", rows)
    con.commit()
    con.close()


def recent_alerts(limit=50, min_tier=("High", "Critical")):
    con = sqlite3.connect(SQLITE_PATH)
    q = ("SELECT ts,node,model,prob,tier FROM predictions "
         f"WHERE tier IN ({','.join('?' * len(min_tier))}) "
         "ORDER BY id DESC LIMIT ?")
    cur = con.execute(q, (*min_tier, limit))
    rows = [dict(zip(["ts", "node", "model", "prob", "tier"], r)) for r in cur.fetchall()]
    con.close()
    return rows


class InfluxSink:
    def __init__(self):
        self.client = None
        try:
            from influxdb_client import InfluxDBClient
            self.client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            self.write_api = self.client.write_api()
        except Exception as e:  # influx optional
            print(f"[influx] disabled: {e}")

    def write(self, node, features_dict, lat, lon):
        if not self.client:
            return
        from influxdb_client import Point
        p = Point("sensor").tag("node", str(node)).tag("lat", lat).tag("lon", lon)
        for k, v in features_dict.items():
            p = p.field(k, float(v))
        try:
            self.write_api.write(bucket=INFLUX_BUCKET, record=p)
        except Exception:
            pass
