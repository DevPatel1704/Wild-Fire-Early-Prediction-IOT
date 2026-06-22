"""Layer 6: virtual IoT environment for TESTING.

Spins up 100 virtual sensor nodes (one per graph node) and streams realistic
top-10 feature readings to Kafka on a fixed tick. The physics matches the
synthetic generator so the live forecast behaves sensibly. A few nodes are
seeded as 'hot' to exercise the High/Critical alert path.

Run:  python -m iot.simulator --tick 1.0 --hot 5
"""
from __future__ import annotations
import argparse
import json
import math
import time
import datetime as dt
import numpy as np
from config import KAFKA_BOOTSTRAP, KAFKA_TOPIC, N_NODES, TOP_10_FEATURES
from iot.graph import save_graph, load_graph


def reading(node, lat, lon, doy, hot=False, rng=None):
    rng = rng or np.random.default_rng()
    season = math.sin((doy - 80) / 365 * 2 * math.pi)
    dry = 0.9 if hot else 0.0
    swvl1 = float(np.clip(0.35 - 0.18 * season - 0.25 * dry + rng.normal(0, 0.04), 0, 1))
    vals = {
        "swvl1": swvl1,
        "mn2t": float(5 + 18 * season + 8 * dry + rng.normal(0, 2)),
        "lgws": float(abs(rng.normal(4 + 2 * season + 4 * dry, 1.5))),
        "pev": float(max(0, rng.normal(2 + 3 * season + 3 * dry, 1))),
        "DOY": float(doy),
        "gwd": float(rng.uniform(0, 360)),
        "blh": float(max(50, rng.normal(800 + 600 * season, 150))),
        "mgws": float(abs(rng.normal(6 + 2 * season + 4 * dry, 1.5))),
        "vilwd": float(rng.normal(0, 1)),
        "swvl2": float(np.clip(swvl1 + rng.normal(0, 0.03), 0, 1)),
    }
    return {
        "node": node, "lat": lat, "lon": lon,
        "ts": dt.datetime.utcnow().isoformat(),
        "features": [vals[f] for f in TOP_10_FEATURES],
    }


def get_producer():
    from kafka import KafkaProducer
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode(),
        retries=5,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick", type=float, default=1.0, help="seconds between full sweeps")
    ap.add_argument("--hot", type=int, default=5, help="nodes seeded as high-risk")
    args = ap.parse_args()

    try:
        coords, _ = load_graph()
    except FileNotFoundError:
        coords, _ = save_graph()

    rng = np.random.default_rng(7)
    hot_nodes = set(rng.choice(N_NODES, size=args.hot, replace=False).tolist())
    producer = get_producer()
    print(f"[sim] {N_NODES} nodes -> {KAFKA_BOOTSTRAP}/{KAFKA_TOPIC}  hot={sorted(hot_nodes)}")

    while True:
        doy = dt.datetime.utcnow().timetuple().tm_yday
        for n, (lat, lon) in enumerate(coords):
            producer.send(KAFKA_TOPIC, reading(n, lat, lon, doy, n in hot_nodes, rng))
        producer.flush()
        time.sleep(args.tick)


if __name__ == "__main__":
    main()
