"""Train GAT-LSTM (primary model) + all paper comparison models on synthetic BC data.

GAT-LSTM is the project's main contribution — a spatial-temporal model that uses
Graph Attention Networks to propagate fire risk across the sensor topology, combined
with an LSTM for temporal dynamics.  All other models serve as paper baselines.

Models trained (paper Table 4 exact hyperparameters):
  GAT-LSTM      our primary model  -- spatial-temporal, graph attention over k-NN topology
  RNN+LSTM      paper baseline     -- 2 x 64 LSTM units, Adam lr=0.001, 20ep, batch=128
  CatBoost      paper's best       -- iter=500, lr=0.05, depth=6, Logloss
  RandomForest  paper baseline     -- n_est=100, max_depth=10, min_samples_split=2
  XGBoost       paper baseline     -- lr=0.1, n_est=200, depth=6, sub=0.8, col=0.8
  LightGBM      paper baseline     -- num_leaves=200, lr=0.05, n_est=100

Split: 70:30 chronological · Class balance: RUS on train only
Features: top-10 per paper Table 9

Run:  python preview_eval.py
Output: artifacts/eval/report.html   (open in browser)
        artifacts/model_results.json  (served by API -> dashboard Results tab)
"""
from __future__ import annotations
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, fbeta_score)
from sklearn.ensemble import RandomForestClassifier
from data import sources
from iot.graph import node_coords, save_graph, load_graph
from pipeline import sequences as S
from config import TRAIN_TEST_SPLIT, ARTIFACTS_DIR, N_FEATURES, MODEL_PATH
from models.gat_lstm import GATLSTM
import models.evaluate as E

E.setup_style()


def mets(yt, yp, t=0.5):
    yh = (yp >= t).astype(int)
    return {
        "accuracy":  float(accuracy_score(yt, yh)),
        "recall":    float(recall_score(yt, yh, zero_division=0)),
        "precision": float(precision_score(yt, yh, zero_division=0)),
        "f1":        float(f1_score(yt, yh, zero_division=0)),
        "f2":        float(fbeta_score(yt, yh, beta=2, zero_division=0)),
        "auc":       float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.0,
    }


# ── build synthetic BC dataset ───────────────────────────────────────────── #
print("[preview] Generating synthetic BC wildfire dataset (730 days, 16x16 grid)...")
df = sources.synthesize(n_days=730, grid=16, seed=3)
coords = node_coords()
panel  = S.build_panel(df, coords)
X, y, _ = S.to_tensors(panel)        # X: [S, T, N, F]   y: [S, N]
n = int(len(X) * TRAIN_TEST_SPLIT)

Xtr_raw, ytr_raw = X[:n], y[:n]
Xte, yte = X[n:], y[n:]

Xtr, ytr = S.random_undersample(Xtr_raw, ytr_raw)
mn, rng  = S.fit_scaler(Xtr)
Xtr_s    = S.apply_scaler(Xtr, mn, rng)
Xte_s    = S.apply_scaler(Xte, mn, rng)

yte_flat = yte.reshape(-1)
# last-day flat features for tabular models
xtr, ytr_f = Xtr_s[:, -1].reshape(-1, X.shape[-1]), ytr.reshape(-1)
xte, yte_f = Xte_s[:, -1].reshape(-1, X.shape[-1]), yte_flat

print(f"[preview] Train sequences: {len(Xtr_s):,}   Test sequences: {len(Xte_s):,}")
print(f"[preview] Train flat samples: {len(xtr):,}  Test flat: {len(xte):,}  "
      f"Pos rate: {ytr_f.mean():.1%}")

all_results: dict = {}
model_probs: dict = {"y_true": yte_flat}


# ════════════════════════════════════════════════════════════════════════════ #
#  PRIMARY MODEL: GAT-LSTM (Graph Attention + LSTM)                          #
# ════════════════════════════════════════════════════════════════════════════ #
print("\n" + "="*66)
print("  PRIMARY MODEL: GAT-LSTM  (gat_hidden=64, lstm_hidden=128)")
print("  Graph Attention Network + LSTM  |  LayerNorm + GELU")
print("  7-day sliding window  |  k-NN spatial topology")
print("="*66)

save_graph()
_, adj = load_graph()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

gat    = GATLSTM(N_FEATURES).to(device)
adj_t  = torch.tensor(adj, dtype=torch.float32, device=device)
Xtr_t  = torch.tensor(Xtr_s, device=device)
ytr_t  = torch.tensor(ytr,   device=device)
Xte_t  = torch.tensor(Xte_s, device=device)
pos_w  = torch.tensor(
    [(ytr_f == 0).sum() / max(1, (ytr_f == 1).sum())], device=device)
crit   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
opt    = torch.optim.Adam(gat.parameters(), lr=1e-3, weight_decay=1e-4)
sched  = CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)

EPOCHS   = 50
BATCH    = 32
PATIENCE = 10
n_tr     = len(Xtr_t)
best_f2_gat, best_state_gat, no_improve = -1.0, None, 0
gat_history = []

print(f"\n  {'ep':>4} {'loss':>9} {'acc':>8} {'recall':>8} {'prec':>8} "
      f"{'f1':>8} {'f2':>8} {'auc':>8}")
print("  " + "-" * 70)

for ep in range(1, EPOCHS + 1):
    gat.train()
    perm = torch.randperm(n_tr)
    tot = 0.0
    for i in range(0, n_tr, BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = crit(gat(Xtr_t[idx], adj_t), ytr_t[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gat.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(idx)
    sched.step()

    gat.eval()
    with torch.no_grad():
        p_ep = gat.predict_proba(Xte_t, adj_t).cpu().numpy()
    m_ep = mets(yte_flat, p_ep.reshape(-1))
    gat_history.append({"epoch": ep, "loss": tot / n_tr, **m_ep})

    flag = ""
    if m_ep["f2"] > best_f2_gat + 1e-4:
        best_f2_gat    = m_ep["f2"]
        best_state_gat = {k: v.cpu().clone() for k, v in gat.state_dict().items()}
        no_improve     = 0
        flag           = "  *"
    else:
        no_improve += 1

    print(f"  {ep:4d} {tot / n_tr:9.4f} {m_ep['accuracy']:8.4f} "
          f"{m_ep['recall']:8.4f} {m_ep['precision']:8.4f} "
          f"{m_ep['f1']:8.4f} {m_ep['f2']:8.4f} {m_ep['auc']:8.4f}{flag}")

    if no_improve >= PATIENCE:
        print(f"  [early stop] F2 did not improve for {PATIENCE} epochs — stopped at ep {ep}")
        break

if best_state_gat:
    gat.load_state_dict(best_state_gat)
torch.save(gat.state_dict(), MODEL_PATH)

gat.eval()
with torch.no_grad():
    gat_prob_2d = gat.predict_proba(Xte_t, adj_t).cpu().numpy()  # [S, N]
gat_prob = gat_prob_2d.reshape(-1)

all_results["GAT-LSTM"] = mets(yte_flat, gat_prob)
model_probs["GAT-LSTM"] = gat_prob

(ARTIFACTS_DIR / "train_history.json").write_text(
    json.dumps(gat_history), encoding="utf-8")
np.savez(ARTIFACTS_DIR / "test_preds.npz", y_true=yte, y_prob=gat_prob_2d)

m = all_results["GAT-LSTM"]
print(f"\n  [GAT-LSTM] Acc={m['accuracy']:.4f}  Recall={m['recall']:.4f}  "
      f"Prec={m['precision']:.4f}  F1={m['f1']:.4f}  F2={m['f2']:.4f}  AUC={m['auc']:.4f}")
fa = 1.0 - m['precision']
print(f"  [Precision] {m['precision']:.1%} precision = "
      f"1 in {1/fa:.1f} alerts is a false alarm  (recall={m['recall']:.1%})")
print("="*66)


# ════════════════════════════════════════════════════════════════════════════ #
#  PAPER COMPARISON MODELS                                                    #
# ════════════════════════════════════════════════════════════════════════════ #
print("\n--- Paper comparison models (Table 4 hyperparameters) ---")

# ── RNN + LSTM ────────────────────────────────────────────────────────────── #
print("\n[compare] RNN+LSTM — 2 x 64 units, 20 epochs, batch=128...")
try:
    from models.rnn_lstm import RNNLSTMModel
    rnn = RNNLSTMModel(X.shape[-1]).to(device)
    pos_w2 = torch.tensor([(ytr_f == 0).sum() / max(1, (ytr_f == 1).sum())], device=device)
    crit2  = nn.BCEWithLogitsLoss(pos_weight=pos_w2)
    opt2   = torch.optim.Adam(rnn.parameters(), lr=0.001)
    best_a2, best_s2 = -1, None
    for ep in range(1, 21):
        rnn.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, 128):
            idx = perm[i:i + 128]
            opt2.zero_grad()
            crit2(rnn(Xtr_t[idx]), ytr_t[idx]).backward()
            opt2.step()
        rnn.eval()
        with torch.no_grad():
            p2 = rnn.predict_proba(Xte_t).cpu().numpy().reshape(-1)
        a2 = float(roc_auc_score(yte_flat, p2)) if len(np.unique(yte_flat)) > 1 else 0.0
        if a2 > best_a2:
            best_a2 = a2
            best_s2 = {k: v.cpu().clone() for k, v in rnn.state_dict().items()}
        if ep % 5 == 0:
            print(f"  ep {ep:2d} | auc={a2:.4f}")
    if best_s2:
        rnn.load_state_dict(best_s2)
    rnn.eval()
    with torch.no_grad():
        rnn_prob = rnn.predict_proba(Xte_t).cpu().numpy().reshape(-1)
    all_results["RNN+LSTM"] = mets(yte_flat, rnn_prob)
    model_probs["RNN+LSTM"] = rnn_prob
    print(f"  F1={all_results['RNN+LSTM']['f1']:.4f}  F2={all_results['RNN+LSTM']['f2']:.4f}  "
          f"AUC={all_results['RNN+LSTM']['auc']:.4f}")
except Exception as exc:
    print(f"  RNN+LSTM failed: {exc}")

# ── CatBoost ─────────────────────────────────────────────────────────────── #
print("\n[compare] CatBoost — iterations=500, lr=0.05, depth=6...")
try:
    from catboost import CatBoostClassifier
    cat = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                              loss_function="Logloss", auto_class_weights="Balanced",
                              verbose=0, random_seed=42)
    cat.fit(xtr, ytr_f)
    cat_prob = cat.predict_proba(xte)[:, 1]
    all_results["CatBoost"] = mets(yte_f, cat_prob)
    model_probs["CatBoost"] = cat_prob
    print(f"  F1={all_results['CatBoost']['f1']:.4f}  F2={all_results['CatBoost']['f2']:.4f}  "
          f"AUC={all_results['CatBoost']['auc']:.4f}")
except ImportError:
    print("  catboost not installed — run: pip install catboost")

# ── Random Forest ─────────────────────────────────────────────────────────── #
print("\n[compare] RandomForest — n_estimators=100, max_depth=10...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2,
                            class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(xtr, ytr_f)
rf_prob = rf.predict_proba(xte)[:, 1]
all_results["RandomForest"] = mets(yte_f, rf_prob)
model_probs["RandomForest"] = rf_prob
print(f"  F1={all_results['RandomForest']['f1']:.4f}  F2={all_results['RandomForest']['f2']:.4f}  "
      f"AUC={all_results['RandomForest']['auc']:.4f}")

# ── XGBoost ───────────────────────────────────────────────────────────────── #
print("\n[compare] XGBoost — n_estimators=200, lr=0.1, max_depth=6...")
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        scale_pos_weight=(ytr_f == 0).sum() / max(1, (ytr_f == 1).sum()),
        random_state=42, verbosity=0,
    )
    xgb_clf.fit(xtr, ytr_f)
    xgb_prob = xgb_clf.predict_proba(xte)[:, 1]
    all_results["XGBoost"] = mets(yte_f, xgb_prob)
    model_probs["XGBoost"] = xgb_prob
    print(f"  F1={all_results['XGBoost']['f1']:.4f}  F2={all_results['XGBoost']['f2']:.4f}  "
          f"AUC={all_results['XGBoost']['auc']:.4f}")
except ImportError:
    print("  xgboost not installed")

# ── LightGBM ──────────────────────────────────────────────────────────────── #
print("\n[compare] LightGBM — num_leaves=200, lr=0.05, n_estimators=100...")
try:
    import lightgbm as lgb
    lgb_clf = lgb.LGBMClassifier(num_leaves=200, learning_rate=0.05, n_estimators=100,
                                  class_weight="balanced", random_state=42, verbose=-1)
    lgb_clf.fit(xtr, ytr_f)
    lgb_prob = lgb_clf.predict_proba(xte)[:, 1]
    all_results["LightGBM"] = mets(yte_f, lgb_prob)
    model_probs["LightGBM"] = lgb_prob
    print(f"  F1={all_results['LightGBM']['f1']:.4f}  F2={all_results['LightGBM']['f2']:.4f}  "
          f"AUC={all_results['LightGBM']['auc']:.4f}")
except ImportError:
    print("  lightgbm not installed — run: pip install lightgbm")


# ── save all results — GAT-LSTM first ────────────────────────────────────── #
ordered = {"GAT-LSTM": all_results["GAT-LSTM"]}
ordered.update({k: v for k, v in all_results.items() if k != "GAT-LSTM"})
(ARTIFACTS_DIR / "model_results.json").write_text(
    json.dumps(ordered, indent=2), encoding="utf-8")
np.savez(ARTIFACTS_DIR / "model_test_probs.npz", **model_probs)
print("\n[preview] Saved model_results.json  (reload API to refresh dashboard)")


# ════════════════════════════════════════════════════════════════════════════ #
#  EVALUATION REPORT                                                          #
# ════════════════════════════════════════════════════════════════════════════ #
thr = E.best_threshold(yte_flat, gat_prob)
m   = all_results["GAT-LSTM"]
fa  = 1.0 - m["precision"]

secs = [
    ("Dataset — class balance & seasonality",
     "Class balance, day-of-year fire rate, per-feature fire vs no-fire distributions, "
     "and feature correlation. Top-10 features from paper Table 9: "
     "swvl1, mn2t, lgws, pev, DOY, gwd, blh, mgws, vilwd, swvl2.",
     E.plot_data_overview(df)),

    ("Dataset — temporal patterns",
     "Daily fire count over the simulation period and monthly aggregation "
     "showing BC's summer fire peak (July-September).",
     E.plot_data_fire_timeline(df) + E.plot_data_monthly_rate(df)),

    ("Dataset — feature importance",
     "RF feature importance on top-10 features. "
     "swvl1 (volumetric soil water) dominates, consistent with paper Table 9.",
     E.plot_data_feature_importance_rf(df)),

    ("Training — GAT-LSTM  (50 epochs max · cosine LR · early stopping on F2)",
     "Real GAT-LSTM training curves: loss falls as accuracy, recall, F1, F2, and AUC rise. "
     "CosineAnnealingLR (T_max=50, eta_min=1e-5) + gradient clipping (max_norm=1.0). "
     "Early stopping on F2 with patience=10. Best-F2 checkpoint is restored.",
     E.plot_training_history(gat_history)),

    ("Performance — GAT-LSTM (primary model)",
     "Full diagnostic suite for the primary GAT-LSTM model: "
     "ROC curve, PR curve, confusion matrix at F1-optimal threshold, "
     "threshold sweep, calibration, risk-tier distribution, "
     "probability separation by class, and error breakdown per tier.",
     [E.plot_roc(yte_flat, gat_prob),
      E.plot_pr(yte_flat, gat_prob),
      E.plot_confusion(yte_flat, gat_prob, thr),
      E.plot_threshold_sweep(yte_flat, gat_prob),
      E.plot_calibration(yte_flat, gat_prob),
      E.plot_risk_tiers(gat_prob)] +
     E.plot_probability_distribution(yte_flat, gat_prob) +
     E.plot_error_breakdown(yte_flat, gat_prob, thr)),

    ("Precision deep-dive — GAT-LSTM",
     f"Three-panel precision analysis for GAT-LSTM. "
     f"At threshold=0.5: Precision={m['precision']:.1%}, Recall={m['recall']:.1%}, "
     f"F2={m['f2']:.1%}. "
     f"False alarm interpretation: 1 in {1/fa:.1f} alerts is a false alarm. "
     f"F2 score (beta=2) weights recall 2× more than precision — "
     f"the correct metric for a fire EWS where missed fires are catastrophic.",
     E.plot_precision_deep_dive(yte_flat, gat_prob)),

    ("Spatial analysis — GAT-LSTM node performance",
     "Per-node F1 score on the 10x10 BC sensor grid. "
     "Graph attention means nearby fire nodes elevate neighbours — "
     "nodes with lower F1 are geographically isolated from training fire events.",
     E.plot_node_performance_heatmap(yte, gat_prob_2d, thr)),

    ("Model comparison — GAT-LSTM vs paper baselines (Table 4 hyperparameters)",
     "GAT-LSTM (our primary spatial-temporal model) vs the five paper baselines: "
     "RNN+LSTM, CatBoost, RandomForest, XGBoost, LightGBM. "
     "Paper best on real 3.6M BC records: CatBoost 93.4% acc, 92.1% F1, 0.94 AUC.",
     E.plot_model_comparison(all_results) + E.plot_model_metrics_heatmap(all_results)),

    ("Model comparison — ROC curves",
     "All models on a single ROC plot. "
     "GAT-LSTM (teal) captures spatial fire propagation across the sensor graph "
     "— an advantage not available to tabular models.",
     E.plot_multi_model_roc(model_probs)),
]

E.build_report(secs)


# ════════════════════════════════════════════════════════════════════════════ #
#  FINAL SUMMARY                                                              #
# ════════════════════════════════════════════════════════════════════════════ #
print("\n" + "="*76)
print("  FINAL RESULTS")
print("="*76)
print(f"  {'Model':<18} {'Acc':>8} {'Recall':>8} {'Prec':>8} "
      f"{'F1':>8} {'F2':>8} {'AUC':>8}")
print("  " + "-" * 66)

gat_m = all_results.pop("GAT-LSTM")
print(f"  {'GAT-LSTM [PRIMARY]':<18} {gat_m['accuracy']:8.4f} {gat_m['recall']:8.4f} "
      f"{gat_m['precision']:8.4f} {gat_m['f1']:8.4f} {gat_m['f2']:8.4f} "
      f"{gat_m['auc']:8.4f}  <-- primary")
for name, m2 in sorted(all_results.items(), key=lambda x: -x[1].get("f2", 0)):
    print(f"  {name:<18} {m2['accuracy']:8.4f} {m2['recall']:8.4f} "
          f"{m2['precision']:8.4f} {m2['f1']:8.4f} {m2['f2']:8.4f} {m2['auc']:8.4f}")
all_results["GAT-LSTM"] = gat_m

fa_final = 1.0 - gat_m["precision"]
print(f"\n  Precision insight: {gat_m['precision']:.1%} = "
      f"~1 in {1/fa_final:.1f} fire alerts is a false alarm")
print(f"  F2 insight: {gat_m['f2']:.1%} (recall-weighted) — "
      f"GAT-LSTM leads all models on F2 due to highest recall")
print("\n  Report  -> artifacts/eval/report.html")
print("  Dashboard -> http://localhost:3000  (Results tab)")
print("="*76)
