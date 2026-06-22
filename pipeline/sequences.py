"""Layer 2->4: turn the tabular dataset into GAT-LSTM tensors.

Steps (paper-faithful where it matters):
  1. snap every grid cell to its nearest virtual node (10x10 grid)
  2. aggregate to one record per (node, date) over the top-10 features
  3. build sliding windows of SEQ_LEN days -> X [S, T, N, F], y [S, N]
  4. Random Undersampling on windows to fight class imbalance
  5. Min-Max scaling fit on TRAIN ONLY (no leakage), applied to test
  6. 70:30 chronological train/test split
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from config import (PROCESSED_DIR, TOP_10_FEATURES, SEQ_LEN, N_NODES,
                    TRAIN_TEST_SPLIT, SCALER_PATH)
from iot.graph import node_coords


def _nearest_node(lat, lon, coords):
    c = np.array(coords)
    d = (c[:, 0] - lat) ** 2 + (c[:, 1] - lon) ** 2
    return int(d.argmin())


def build_panel(df: pd.DataFrame, coords) -> pd.DataFrame:
    df = df.copy()
    df["node"] = [_nearest_node(la, lo, coords) for la, lo in zip(df["lat"], df["lon"])]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    agg = {f: "mean" for f in TOP_10_FEATURES}
    agg["fire"] = "max"
    panel = df.groupby(["date", "node"], as_index=False).agg(agg)
    return panel


def to_tensors(panel: pd.DataFrame):
    dates = np.sort(panel["date"].unique())
    F = len(TOP_10_FEATURES)
    # dense [D, N, F] feature cube + [D, N] labels, forward-filled gaps
    feat = np.zeros((len(dates), N_NODES, F), dtype=np.float32)
    lab = np.zeros((len(dates), N_NODES), dtype=np.float32)
    didx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    for _, r in panel.iterrows():
        di = didx[r["date"]]
        feat[di, int(r["node"])] = r[TOP_10_FEATURES].to_numpy(dtype=np.float32)
        lab[di, int(r["node"])] = r["fire"]

    X, y = [], []
    for t in range(SEQ_LEN, len(dates)):
        X.append(feat[t - SEQ_LEN:t])     # [T, N, F]
        y.append(lab[t])                  # [N]
    return np.array(X), np.array(y), dates[SEQ_LEN:]


def random_undersample(X, y, ratio=1.0, seed=7):
    """Balance windows by whether they contain ANY fire node."""
    rng = np.random.default_rng(seed)
    pos = np.where(y.max(axis=1) > 0)[0]
    neg = np.where(y.max(axis=1) == 0)[0]
    if len(pos) == 0:           # nothing to balance against; keep as-is
        return X, y
    keep_neg = rng.choice(neg, size=min(len(neg), int(len(pos) * ratio)), replace=False)
    idx = np.sort(np.concatenate([pos, keep_neg]))
    return X[idx], y[idx]


def fit_scaler(X):
    flat = X.reshape(-1, X.shape[-1])
    mn, mx = flat.min(axis=0), flat.max(axis=0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    SCALER_PATH.write_text(json.dumps({"min": mn.tolist(), "range": rng.tolist()}))
    return mn, rng


def apply_scaler(X, mn=None, rng=None):
    if mn is None:
        s = json.loads(SCALER_PATH.read_text())
        mn, rng = np.array(s["min"]), np.array(s["range"])
    return ((X - mn) / rng).astype(np.float32)


def prepare():
    coords = node_coords()
    df = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    panel = build_panel(df, coords)
    X, y, dates = to_tensors(panel)

    n_train = int(len(X) * TRAIN_TEST_SPLIT)          # chronological 70:30
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]
    Xtr, ytr = random_undersample(Xtr, ytr)            # RUS on train only

    mn, rng = fit_scaler(Xtr)                          # scaler fit on train only
    Xtr, Xte = apply_scaler(Xtr, mn, rng), apply_scaler(Xte, mn, rng)
    print(f"[prep] train {Xtr.shape} test {Xte.shape} pos_rate_tr={ytr.mean():.3f}")
    return (Xtr, ytr), (Xte, yte)


if __name__ == "__main__":
    prepare()
