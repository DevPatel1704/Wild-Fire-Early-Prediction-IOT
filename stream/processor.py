"""Layer 6->7: the live stream processor (LIVE box in the diagram).

Consumes sensor readings from Kafka, keeps a rolling SEQ_LEN window of frames,
and on the architecture's timing contract:
  * every 60 s  -> XGBoost fast alert  (per latest frame)
  * every 7 min -> GAT-LSTM forecast   (per window)  [main model]
Results are written to SQLite, raw readings to InfluxDB, and predictions are
republished to the `predictions` Kafka topic for the API to fan out.

Run:  python -m stream.processor
"""
from __future__ import annotations
import json
import time
import datetime as dt
from collections import deque
import numpy as np
from config import (KAFKA_BOOTSTRAP, KAFKA_TOPIC, N_NODES, N_FEATURES, SEQ_LEN,
                    XGB_ALERT_EVERY_S, GATLSTM_FORECAST_EVERY_S)
from models.infer import RiskEngine
from api.db import init_db, save_predictions, InfluxSink, recent_alerts  # noqa: F401
from iot.graph import load_graph

PRED_TOPIC = "predictions"


def consumer_producer():
    from kafka import KafkaConsumer, KafkaProducer
    c = KafkaConsumer(
        KAFKA_TOPIC, bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="latest", group_id="stream-processor",
        consumer_timeout_ms=200,
    )
    p = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP,
                      value_serializer=lambda v: json.dumps(v).encode())
    return c, p


def publish(producer, kind, preds):
    payload = {"kind": kind, "ts": dt.datetime.utcnow().isoformat(), "preds": preds}
    producer.send(PRED_TOPIC, payload)
    producer.flush()
    rows = [(payload["ts"], p["node"], p["model"], p["prob"], p["tier"]) for p in preds]
    save_predictions(rows)


def main():
    init_db()
    coords, _ = load_graph()
    engine = RiskEngine()
    influx = InfluxSink()
    from config import TOP_10_FEATURES

    latest = np.zeros((N_NODES, N_FEATURES), dtype=np.float32)
    window = deque(maxlen=SEQ_LEN)
    consumer, producer = consumer_producer()
    print(f"[stream] consuming {KAFKA_TOPIC}; XGB every {XGB_ALERT_EVERY_S}s, "
          f"GAT-LSTM every {GATLSTM_FORECAST_EVERY_S}s")

    last_xgb = last_gat = 0.0
    while True:
        for msg in consumer:
            r = msg.value
            latest[r["node"]] = r["features"]
            influx.write(r["node"], dict(zip(TOP_10_FEATURES, r["features"])),
                         r["lat"], r["lon"])

        now = time.time()
        # snapshot a frame into the window roughly each forecast tick window
        if not window or now - getattr(main, "_lastframe", 0) >= 5:
            window.append(latest.copy())
            main._lastframe = now

        if now - last_xgb >= XGB_ALERT_EVERY_S:
            preds = engine.xgb_alert(latest.copy())
            if preds:
                publish(producer, "alert", preds)
                hi = [p for p in preds if p["tier"] in ("High", "Critical")]
                print(f"[stream] XGB alert  -> {len(hi)} High/Critical nodes")
            last_xgb = now

        if now - last_gat >= GATLSTM_FORECAST_EVERY_S and len(window) == SEQ_LEN:
            preds = engine.gatlstm_forecast(np.array(window))
            if preds:
                publish(producer, "forecast", preds)
                print("[stream] GAT-LSTM forecast published")
            last_gat = now

        time.sleep(0.5)


if __name__ == "__main__":
    main()
