"""Export the synthetic wildfire dataset to CSV files (full + train/test splits)."""
from data import sources
from pipeline import sequences as S
from iot.graph import node_coords
from config import TRAIN_TEST_SPLIT, TOP_10_FEATURES
import pandas as pd
import numpy as np

print("[export] Generating synthetic BC wildfire dataset (730 days, 16x16 grid)...")
df = sources.synthesize(n_days=730, grid=16, seed=3)

print(f"  Shape        : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Date range   : {df['date'].min().date()}  ->  {df['date'].max().date()}")
print(f"  Grid points  : {df.groupby(['lat','lon']).ngroups} unique lat/lon locations")
print(f"  Fire rate    : {df['fire'].mean():.2%}  ({df['fire'].sum():,} fire records)")
print(f"  Columns      : {list(df.columns)}")

# ── 1. Full raw dataset ───────────────────────────────────────
df['date'] = df['date'].astype(str)
df.to_csv("data/wildfire_dataset_full.csv", index=False)
print(f"\n[export] Saved: data/wildfire_dataset_full.csv  ({len(df):,} rows)")

# ── 2. Node-level panel (one row per node per day) ────────────
coords = node_coords()
panel = S.build_panel(df, coords)
panel['date'] = panel['date'].astype(str)
panel.to_csv("data/wildfire_dataset_panel.csv", index=False)
print(f"[export] Saved: data/wildfire_dataset_panel.csv  ({len(panel):,} rows, 100 nodes x ~730 days)")

# ── 3. Train split (flat, last-day features for tabular models) ─
df2 = sources.synthesize(n_days=730, grid=16, seed=3)
coords2 = node_coords()
panel2 = S.build_panel(df2, coords2)
X, y, dates = S.to_tensors(panel2)

n = int(len(X) * TRAIN_TEST_SPLIT)
Xtr_raw, ytr_raw = X[:n], y[:n]
Xte, yte = X[n:], y[n:]

Xtr, ytr = S.random_undersample(Xtr_raw, ytr_raw)
mn, rng  = S.fit_scaler(Xtr)
Xtr_s    = S.apply_scaler(Xtr, mn, rng)
Xte_s    = S.apply_scaler(Xte, mn, rng)

# flat (last day of each window) for tabular inspection
xtr_flat = Xtr_s[:, -1].reshape(-1, X.shape[-1])
ytr_flat = ytr.reshape(-1)
xte_flat = Xte_s[:, -1].reshape(-1, X.shape[-1])
yte_flat = yte.reshape(-1)

train_df = pd.DataFrame(xtr_flat, columns=[f + "_scaled" for f in TOP_10_FEATURES])
train_df['fire'] = ytr_flat.astype(int)
train_df.to_csv("data/wildfire_train.csv", index=False)
print(f"[export] Saved: data/wildfire_train.csv  ({len(train_df):,} rows, RUS balanced, scaled)")

test_df = pd.DataFrame(xte_flat, columns=[f + "_scaled" for f in TOP_10_FEATURES])
test_df['fire'] = yte_flat.astype(int)
test_df.to_csv("data/wildfire_test.csv", index=False)
print(f"[export] Saved: data/wildfire_test.csv   ({len(test_df):,} rows, unbalanced, scaled)")

# ── 4. Summary stats ──────────────────────────────────────────
print("\n" + "="*55)
print("  DATASET SUMMARY")
print("="*55)
print(f"  Full dataset     : {len(df):,} rows  (730 days x 256 grid pts)")
print(f"  Node panel       : {len(panel):,} rows  (730 days x 100 nodes)")
print(f"  Train sequences  : {len(Xtr_s):,}  (70% dates, RUS 1:1)")
print(f"  Test  sequences  : {len(Xte_s):,}  (30% dates, unbalanced)")
print(f"  Train flat rows  : {len(train_df):,}")
print(f"  Test  flat rows  : {len(test_df):,}")
print(f"  Train fire rate  : {ytr_flat.mean():.1%}  (after RUS)")
print(f"  Test  fire rate  : {yte_flat.mean():.1%}  (real distribution)")
print(f"  Features         : {list(TOP_10_FEATURES)}")
print("="*55)

print("\n[export] Sample rows from full dataset:")
print(df.head(8).to_string(index=False))
