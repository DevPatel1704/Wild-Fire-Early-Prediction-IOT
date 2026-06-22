"""Layer 1->2: build the integrated, labelled dataset.

Mirrors Figure 1 of the paper: load ERA5 grid + fire points (NFDB/FIRMS),
label each grid cell/day as fire(1)/no-fire(0) by Haversine proximity to a
documented fire on the same date, drop obvious water cells, write processed
parquet. Falls back to synthetic data when raw files are absent.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import PROCESSED_DIR, TOP_10_FEATURES
from data import sources


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def label_by_proximity(grid: pd.DataFrame, fires: pd.DataFrame, radius_km=10.0):
    """Assign fire=1 to a grid cell if a fire point is within radius on same date."""
    grid = grid.copy()
    grid["date"] = pd.to_datetime(grid["date"]).dt.normalize()
    fires = fires.copy()
    fires["date"] = pd.to_datetime(fires["date"]).dt.normalize()
    grid["fire"] = 0
    for d, fday in fires.groupby("date"):
        m = grid["date"] == d
        if not m.any():
            continue
        gl = grid.loc[m]
        flat, flon = fday["lat"].to_numpy(), fday["lon"].to_numpy()
        hit = np.zeros(len(gl), dtype=bool)
        for la, lo in zip(flat, flon):
            hit |= haversine_km(gl["lat"].to_numpy(), gl["lon"].to_numpy(), la, lo) <= radius_km
        grid.loc[m, "fire"] = hit.astype(int)
    return grid


def build(radius_km=10.0) -> pd.DataFrame:
    try:
        era5 = sources.load_era5()
        fires = pd.concat([sources.load_nfdb(), sources.load_firms()], ignore_index=True)
        df = label_by_proximity(era5, fires, radius_km)
        print(f"[build] real sources integrated: {len(df):,} rows")
    except FileNotFoundError as e:
        print(f"[build] raw source missing ({e}); using synthetic generator")
        df = sources.synthesize()

    # keep only what the model needs + keys
    keep = ["date", "lat", "lon", "fire"] + TOP_10_FEATURES
    df = df[[c for c in keep if c in df.columns]].dropna().reset_index(drop=True)
    out = PROCESSED_DIR / "dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"[build] wrote {out}  shape={df.shape}  fire_rate={df['fire'].mean():.4f}")
    return df


if __name__ == "__main__":
    build()
