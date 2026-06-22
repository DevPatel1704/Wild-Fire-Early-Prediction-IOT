"""Kafka-free fire-scenario demo for the live dashboard.

Pushes fire predictions directly to the API (/demo/push) every few seconds
so the dashboard shows live risk changes without needing Docker/Kafka.

GAT-LSTM  (kind=forecast) and XGBoost (kind=alert) predictions are sent
separately so each model's page on the dashboard receives its own data.

Differences between the two model outputs:
  GAT-LSTM  - spatially-aware: fire nodes AND their k-NN neighbours are
              elevated (graph attention propagates risk across the topology)
  XGBoost   - tabular, per-node: only the fire nodes themselves show High/
              Critical (no graph structure, no spatial spreading)

3 scenarios cycle automatically (~24 s each):
  A) NE Corner fire      - compact cluster in the north-east
  B) Diagonal corridor   - fire front sweeping diagonally across BC
  C) Multi-ignition      - scattered simultaneous ignition points

Run (while API is up):
    python demo_fire.py

Watch the dashboard at http://localhost:3000
"""
from __future__ import annotations
import datetime as dt
import json
import time
import urllib.request

import numpy as np

from config import N_NODES, risk_tier
from iot.graph import load_graph

API    = "http://localhost:8000"
TICK_S = 3               # seconds between prediction pushes
TICKS_PER_SCENARIO = 8   # ~24 s per scenario before rotating

# --------------------------------------------------------------------------- #
# Fire scenarios                                                              #
# --------------------------------------------------------------------------- #
SCENARIOS = [
    (
        "Scenario A  NE Corner fire",
        lambda n: (n // 10) >= 7 and (n % 10) >= 7,    # top-right 3×3 block
    ),
    (
        "Scenario B  Diagonal fire corridor",
        lambda n: abs((n // 10) - (n % 10)) <= 1,       # main diagonal band
    ),
    (
        "Scenario C  Multi-ignition (scattered)",
        lambda n: n in {4, 14, 22, 35, 47, 56, 63, 76, 89, 95},
    ),
]


def _neighbours(n: int) -> set[int]:
    """8-connected grid neighbours of node n on the 10×10 grid."""
    row, col = n // 10, n % 10
    nbrs: set[int] = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r2, c2 = row + dr, col + dc
            if 0 <= r2 < 10 and 0 <= c2 < 10:
                nbrs.add(r2 * 10 + c2)
    return nbrs


def make_gatlstm_preds(coords: list, hot_set: set[int], rng) -> list[dict]:
    """GAT-LSTM: fire nodes + neighbours elevated (spatial graph spreading)."""
    warm_set: set[int] = set()
    for n in hot_set:
        warm_set |= _neighbours(n)
    warm_set -= hot_set

    preds = []
    for n, (lat, lon) in enumerate(coords):
        if n in hot_set:
            prob = float(np.clip(rng.normal(0.70, 0.14), 0.31, 0.97))
        elif n in warm_set:
            # graph attention propagates partial risk to neighbours
            prob = float(np.clip(rng.normal(0.19, 0.07), 0.07, 0.34))
        else:
            prob = float(np.clip(rng.normal(0.025, 0.012), 0.002, 0.07))
        label, colour = risk_tier(prob)
        preds.append({"node": n, "lat": lat, "lon": lon,
                      "prob": prob, "tier": label, "colour": colour,
                      "model": "gat-lstm"})
    return preds


def make_xgb_preds(coords: list, hot_set: set[int], rng) -> list[dict]:
    """XGBoost: only fire nodes are elevated — no spatial spreading."""
    preds = []
    for n, (lat, lon) in enumerate(coords):
        if n in hot_set:
            # tabular model captures local fire features
            prob = float(np.clip(rng.normal(0.60, 0.16), 0.20, 0.93))
        else:
            # non-fire nodes stay Low; XGBoost sees no graph context
            prob = float(np.clip(rng.normal(0.030, 0.018), 0.002, 0.09))
        label, colour = risk_tier(prob)
        preds.append({"node": n, "lat": lat, "lon": lon,
                      "prob": prob, "tier": label, "colour": colour,
                      "model": "xgboost"})
    return preds


def push(payload: dict) -> None:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{API}/demo/push", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    urllib.request.urlopen(req, timeout=5)


def main() -> None:
    coords, _ = load_graph()
    rng = np.random.default_rng()

    print(f"[demo] sending predictions to {API}/demo/push every {TICK_S}s")
    print("[demo] Dashboard  -> http://localhost:3000")
    print("[demo]   GAT-LSTM tab: spatial risk with neighbour spreading")
    print("[demo]   XGBoost  tab: tabular risk, fire nodes only")
    print("[demo] Press Ctrl+C to stop\n")

    tick     = 0
    hot_set: set[int] = set()
    name     = ""

    while True:
        # ── rotate scenario ──────────────────────────────────────────────── #
        if tick % TICKS_PER_SCENARIO == 0:
            idx  = (tick // TICKS_PER_SCENARIO) % len(SCENARIOS)
            name, hot_fn = SCENARIOS[idx]
            hot_set = {n for n in range(N_NODES) if hot_fn(n)}
            warm_n  = len({nb for n in hot_set for nb in _neighbours(n)} - hot_set)
            print(f"\n[demo] *** {name} ***")
            print(f"[demo]     fire nodes={len(hot_set)}  "
                  f"spreading neighbours (GAT-LSTM only)={warm_n}")

        ts = dt.datetime.utcnow().isoformat()

        # ── every tick: send GAT-LSTM forecast ───────────────────────────── #
        gat_preds = make_gatlstm_preds(coords, hot_set, rng)
        try:
            push({"kind": "forecast", "ts": ts, "preds": gat_preds})
        except Exception as exc:
            print(f"[demo] GAT-LSTM push failed: {exc}  (is the API running?)")

        # ── every other tick: also send XGBoost alert ─────────────────────── #
        if tick % 2 == 0:
            xgb_preds = make_xgb_preds(coords, hot_set, rng)
            try:
                push({"kind": "alert", "ts": ts, "preds": xgb_preds})
            except Exception as exc:
                print(f"[demo] XGBoost push failed: {exc}")

        gat_hi = sum(1 for p in gat_preds if p["tier"] in ("High", "Critical"))
        print(f"[demo] tick {tick:4d} | GAT-LSTM Hi/Crit={gat_hi:3d}  "
              f"(XGBoost sent every 2nd tick)")

        tick += 1
        time.sleep(TICK_S)


if __name__ == "__main__":
    main()
