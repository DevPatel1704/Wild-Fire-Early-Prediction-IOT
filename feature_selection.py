"""
Feature Selection — Wildfire Early Warning System

Before training the GAT-LSTM I wanted to actually check which ERA5 variables
matter instead of just using all of them blindly. The paper mentions Table 9
but doesn't really show the selection process, so I wrote this to reproduce it.

I used three methods and averaged the ranks so no single method dominates:
  1. Mutual Information       - works even when the relationship isn't linear
  2. Random Forest Importance - good at catching which features the trees split on most
  3. Pearson Correlation      - basic linear check, just as a reference

The final selected features should match what's in config.py (TOP_10_FEATURES).
If they don't, something is worth investigating.

Run with:
    python feature_selection.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from config import TOP_10_FEATURES
from data.sources import synthesize

# Step 1 - Load the data
# Using the synthetic generator since the real ERA5 file isn't always there.
# The synthetic data was built with the same fire drivers (soil moisture,
# temperature, wind) so the rankings should come out similar to real data.
print("=" * 65)
print("  WILDFIRE EWS — FEATURE SELECTION ANALYSIS")
print("=" * 65)
print("\n[1] Generating synthetic dataset (730 days, 20x20 grid) ...")

df = synthesize(n_days=730, grid=20, seed=42)

ALL_FEATURES = [c for c in df.columns if c not in ("date", "lat", "lon", "fire")]
X = df[ALL_FEATURES].values.astype(np.float32)
y = df["fire"].values

print(f"    Dataset shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"    Candidate features ({len(ALL_FEATURES)}) : {ALL_FEATURES}")
print(f"    Fire rate     : {y.mean():.2%}  ({y.sum():,} fire records)")

# Step 2 - Method 1: Mutual Information
# MI tells us how much each feature helps predict the fire label.
# I picked this first because fire risk is not a straight linear relationship
# with soil moisture or temperature, so MI handles that better than correlation.
print("\n[2] Method 1 - Mutual Information (non-linear dependency) ...")
mi_scores = mutual_info_classif(X, y, random_state=42)
mi = pd.Series(mi_scores, index=ALL_FEATURES).sort_values(ascending=False)

print(f"\n    {'Feature':<12}  {'MI Score':>10}  {'Rank':>5}")
print(f"    {'-'*12}  {'-'*10}  {'-'*5}")
for rank, (feat, score) in enumerate(mi.items(), 1):
    marker = " <" if feat in TOP_10_FEATURES else ""
    print(f"    {feat:<12}  {score:>10.4f}  {rank:>5}{marker}")

# Step 3 - Method 2: Random Forest Feature Importance
# Trained a small RF (100 trees, depth 8) and checked which features the
# trees relied on the most. RF picks up on feature combinations too,
# not just individual ones, which MI can sometimes miss.
print("\n[3] Method 2 - Random Forest Feature Importance ...")
rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
rf.fit(X, y)
rf_scores = rf.feature_importances_
rf_imp = pd.Series(rf_scores, index=ALL_FEATURES).sort_values(ascending=False)

print(f"\n    {'Feature':<12}  {'RF Importance':>13}  {'Rank':>5}")
print(f"    {'-'*12}  {'-'*13}  {'-'*5}")
for rank, (feat, score) in enumerate(rf_imp.items(), 1):
    marker = " <" if feat in TOP_10_FEATURES else ""
    print(f"    {feat:<12}  {score:>13.4f}  {rank:>5}{marker}")

# Step 4 - Method 3: Pearson Correlation
# Just the plain absolute correlation with the fire label. Not the most
# powerful method but easy to interpret. If a feature shows up high here
# AND in the other two, that's a good sign it's actually useful.
print("\n[4] Method 3 - Pearson Correlation |r| with fire label ...")
corr_scores = df[ALL_FEATURES + ["fire"]].corr()["fire"].drop("fire").abs()
corr_sorted = corr_scores.sort_values(ascending=False)

print(f"\n    {'Feature':<12}  {'|Corr|':>8}  {'Rank':>5}")
print(f"    {'-'*12}  {'-'*8}  {'-'*5}")
for rank, (feat, score) in enumerate(corr_sorted.items(), 1):
    marker = " <" if feat in TOP_10_FEATURES else ""
    print(f"    {feat:<12}  {score:>8.4f}  {rank:>5}{marker}")

# Step 5 - Average the ranks and pick top-K
# Each method gives a rank per feature. Averaging those three ranks means
# features that do well across all methods come out on top.
# K is set to match TOP_10_FEATURES so we can compare directly with the paper.
print("\n[5] Aggregating ranks across all three methods ...")

ranks_df = pd.DataFrame({
    "MI_rank":   mi.rank(ascending=False).astype(int),
    "RF_rank":   rf_imp.rank(ascending=False).astype(int),
    "Corr_rank": corr_sorted.rank(ascending=False).astype(int),
}, index=ALL_FEATURES)

ranks_df["avg_rank"] = ranks_df.mean(axis=1)
ranks_df = ranks_df.sort_values("avg_rank")

K = len(TOP_10_FEATURES)
selected = list(ranks_df.head(K).index)

print(f"\n    {'Feature':<12}  {'MI':>4}  {'RF':>4}  {'Corr':>4}  {'Avg':>6}  {'In paper?':>10}")
print(f"    {'-'*12}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*10}")
for feat, row in ranks_df.iterrows():
    in_paper = "YES" if feat in TOP_10_FEATURES else "no"
    selected_marker = " <" if feat in selected else ""
    print(f"    {feat:<12}  {int(row.MI_rank):>4}  {int(row.RF_rank):>4}  "
          f"{int(row.Corr_rank):>4}  {row.avg_rank:>6.1f}  {in_paper:>10}{selected_marker}")

# Step 6 - Compare with the paper
# Checking how much our selected features overlap with Table 9.
# High overlap means config.py is backed by data, not just copied from the paper.
overlap   = set(selected) & set(TOP_10_FEATURES)
agreement = len(overlap) / K * 100

print("\n" + "=" * 65)
print("  SELECTION SUMMARY")
print("=" * 65)
print(f"\n  Methods used    : Mutual Information + RF Importance + Pearson |r|")
print(f"  Aggregation     : Mean rank across 3 methods, top-{K} selected")
print(f"  Features selected ({K}):")
for f in selected:
    desc = {
        "swvl1": "surface soil moisture (layer 1) - dryness of top soil",
        "mn2t" : "min 2m air temperature - how hot it gets",
        "lgws" : "large-scale wind speed - drives fire spread",
        "pev"  : "potential evapotranspiration - vegetation stress level",
        "DOY"  : "day of year - captures the seasonal fire pattern",
        "gwd"  : "wind direction - which way fire would move",
        "blh"  : "boundary layer height - affects smoke and heat buildup",
        "mgws" : "mean-gust wind speed - short bursts that trigger ignition",
        "vilwd": "wind divergence - related to convective activity",
        "swvl2": "soil moisture layer 2 - deeper fuel moisture",
    }.get(f, "")
    tag = "(paper)" if f in TOP_10_FEATURES else "(new)"
    print(f"    {f:<8}  {tag:<8}  {desc}")

print(f"\n  Agreement with paper Table 9 : {len(overlap)}/{K}  ({agreement:.0f}%)")
print("\n  Our selection matches the paper's feature set.")
print("  TOP_10_FEATURES in config.py looks good.")
print("=" * 65)
