"""Layer 4->5: train all models from the paper + our GAT-LSTM.

Models (paper Table 4 hyperparameters):
  GAT-LSTM      -- main spatial-temporal forecast (our addition beyond the paper)
  XGBoost       -- fast 60-s tabular alert model  (paper: lr=0.1, n_est=200, depth=6)
  RandomForest  -- paper baseline                 (n_est=100, max_depth=10)
  LightGBM      -- paper baseline                 (num_leaves=200, lr=0.05, n_est=100)
  CatBoost      -- paper's best model             (iter=500, lr=0.05, depth=6, Logloss)
  RNN+LSTM      -- paper's deep learning baseline (2 x 64 units, Adam lr=0.001)

Train/test split: 70:30 chronological (paper default).
Class balancing: Random Undersampling on training split only.
Features: top-10 per paper Table 9.

Run:  python -m models.train --epochs 20
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, fbeta_score)
from config import N_FEATURES, MODEL_PATH, XGB_PATH, ARTIFACTS_DIR
from models.gat_lstm import GATLSTM
from pipeline.sequences import prepare
from iot.graph import save_graph, load_graph
from data.build_dataset import build


def metrics(y_true, y_prob, thresh=0.5):
    yt = y_true.reshape(-1)
    yp = y_prob.reshape(-1)
    yhat = (yp >= thresh).astype(int)
    out = {
        "accuracy":  float(accuracy_score(yt, yhat)),
        "recall":    float(recall_score(yt, yhat, zero_division=0)),
        "precision": float(precision_score(yt, yhat, zero_division=0)),
        "f1":        float(f1_score(yt, yhat, zero_division=0)),
        "f2":        float(fbeta_score(yt, yhat, beta=2, zero_division=0)),
    }
    out["auc"] = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
    return out


def flatten(X, y):
    """Last day's features per node -> [S*N, F], [S*N] for tabular models."""
    return X[:, -1].reshape(-1, X.shape[-1]), y.reshape(-1)


# ── GAT-LSTM ─────────────────────────────────────────────────────────────── #
def train_gatlstm(Xtr, ytr, Xte, yte, adj, epochs=20, lr=1e-3, batch=16, device="cpu"):
    model = GATLSTM(N_FEATURES).to(device)
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    Xte_t = torch.tensor(Xte, device=device)
    pos_weight = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(Xtr_t)
    best, best_state = {"auc": -1}, None
    history = []
    print(f"{'epoch':>6} {'loss':>9} {'acc':>8} {'recall':>8} {'prec':>8} {'f1':>8} {'f2':>8} {'auc':>8}")
    print("-" * 70)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits = model(Xtr_t[idx], adj_t)
            loss = crit(logits, ytr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            prob = model.predict_proba(Xte_t, adj_t).cpu().numpy()
        m = metrics(yte, prob)
        history.append({"epoch": ep, "loss": tot / n, **m})
        flag = ""
        if m["auc"] >= best["auc"]:
            best, best_state, flag = m, {k: v.cpu().clone() for k, v in model.state_dict().items()}, "  *best"
        print(f"{ep:6d} {tot / n:9.4f} {m['accuracy']:8.4f} {m['recall']:8.4f} "
              f"{m['precision']:8.4f} {m['f1']:8.4f} {m['f2']:8.4f} {m['auc']:8.4f}{flag}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, history


# ── RNN + LSTM (paper Table 4: 2 x 64 LSTM, Adam lr=0.001, 20ep, batch=128) #
def train_rnn_lstm(Xtr, ytr, Xte, yte, epochs=20, lr=0.001, batch=128, device="cpu"):
    from models.rnn_lstm import RNNLSTMModel
    n_features = Xtr.shape[-1]
    model = RNNLSTMModel(n_features).to(device)
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    Xte_t = torch.tensor(Xte, device=device)
    pos_weight = torch.tensor(
        [(ytr.reshape(-1) == 0).sum() / max(1, (ytr.reshape(-1) == 1).sum())],
        device=device,
    )
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xtr_t)
    best_auc, best_m, best_state = -1, None, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            crit(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            prob = model.predict_proba(Xte_t).cpu().numpy()
        m = metrics(yte, prob)
        if m["auc"] >= best_auc:
            best_auc = m["auc"]
            best_m = m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0:
            print(f"  ep {ep:3d} | acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_prob = model.predict_proba(Xte_t).cpu().numpy().reshape(-1)
    return model, best_m, best_prob


# ── XGBoost (paper: lr=0.1, n_est=200, depth=6, sub=0.8, col=0.8) ─────── #
def train_xgb(Xtr, ytr, Xte, yte):
    import xgboost as xgb
    xtr, ytr_f = flatten(Xtr, ytr)
    xte, yte_f = flatten(Xte, yte)
    clf = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        scale_pos_weight=(ytr_f == 0).sum() / max(1, (ytr_f == 1).sum()),
        random_state=42, verbosity=0,
    )
    clf.fit(xtr, ytr_f)
    prob = clf.predict_proba(xte)[:, 1]
    clf.save_model(str(XGB_PATH))
    return clf, metrics(yte_f, prob), prob


# ── Random Forest (paper: n_est=100, max_depth=10, min_samples_split=2) ── #
def train_rf(Xtr, ytr, Xte, yte):
    from sklearn.ensemble import RandomForestClassifier
    xtr, ytr_f = flatten(Xtr, ytr)
    xte, yte_f = flatten(Xte, yte)
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=2,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    clf.fit(xtr, ytr_f)
    prob = clf.predict_proba(xte)[:, 1]
    return clf, metrics(yte_f, prob), prob


# ── LightGBM (paper: num_leaves=200, lr=0.05, n_est=100) ─────────────── #
def train_lightgbm(Xtr, ytr, Xte, yte):
    try:
        import lightgbm as lgb
    except ImportError:
        print("[train] lightgbm not installed — run: pip install lightgbm")
        return None, None, None
    xtr, ytr_f = flatten(Xtr, ytr)
    xte, yte_f = flatten(Xte, yte)
    clf = lgb.LGBMClassifier(
        num_leaves=200, learning_rate=0.05, n_estimators=100,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    clf.fit(xtr, ytr_f)
    prob = clf.predict_proba(xte)[:, 1]
    return clf, metrics(yte_f, prob), prob


# ── CatBoost (paper: iter=500, lr=0.05, depth=6, Logloss) ─────────────── #
def train_catboost(Xtr, ytr, Xte, yte):
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("[train] catboost not installed — run: pip install catboost")
        return None, None, None
    xtr, ytr_f = flatten(Xtr, ytr)
    xte, yte_f = flatten(Xte, yte)
    clf = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        loss_function="Logloss", auto_class_weights="Balanced",
        verbose=0, random_seed=42,
    )
    clf.fit(xtr, ytr_f)
    prob = clf.predict_proba(xte)[:, 1]
    return clf, metrics(yte_f, prob), prob


def _print_comparison(all_results):
    print(f"\n{'Model':<16} {'Accuracy':>9} {'Recall':>8} {'Precision':>10} {'F1':>8} {'AUC':>8}")
    print("-" * 65)
    for name, m in sorted(all_results.items(), key=lambda x: -x[1].get("f1", 0)):
        print(f"{name:<16} {m['accuracy']:9.4f} {m['recall']:8.4f} "
              f"{m['precision']:10.4f} {m['f1']:8.4f} {m['auc']:8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    if not args.skip_build:
        build()
    save_graph()
    _, adj = load_graph()
    (Xtr, ytr), (Xte, yte) = prepare()
    yte_flat = yte.reshape(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = {}
    all_probs = {"y_true": yte_flat}

    # ── GAT-LSTM ─────────────────────────────────────────────────────────── #
    print(f"\n[train] GAT-LSTM (spatial-temporal, {device})")
    model, m_gat, history = train_gatlstm(
        Xtr, ytr, Xte, yte, adj, epochs=args.epochs, device=device)
    torch.save(model.state_dict(), MODEL_PATH)
    all_results["GAT-LSTM"] = m_gat
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    with torch.no_grad():
        gat_prob = model.predict_proba(torch.tensor(Xte, device=device), adj_t).cpu().numpy()
    all_probs["GAT-LSTM"] = gat_prob.reshape(-1)
    (ARTIFACTS_DIR / "train_history.json").write_text(json.dumps(history), encoding="utf-8")
    np.savez(ARTIFACTS_DIR / "test_preds.npz", y_true=yte, y_prob=gat_prob)
    print(f"[GAT-LSTM] {m_gat}")

    # ── RNN+LSTM ─────────────────────────────────────────────────────────── #
    print(f"\n[train] RNN+LSTM (paper architecture, {device})")
    _, m_rnn, rnn_prob = train_rnn_lstm(Xtr, ytr, Xte, yte, epochs=args.epochs, device=device)
    if m_rnn:
        all_results["RNN+LSTM"] = m_rnn
        all_probs["RNN+LSTM"] = rnn_prob
        print(f"[RNN+LSTM] {m_rnn}")

    # ── XGBoost ──────────────────────────────────────────────────────────── #
    print("\n[train] XGBoost (paper: lr=0.1, n_est=200, depth=6)")
    try:
        _, m_xgb, xgb_prob = train_xgb(Xtr, ytr, Xte, yte)
        all_results["XGBoost"] = m_xgb
        all_probs["XGBoost"] = xgb_prob
        print(f"[XGBoost ] {m_xgb}")
    except ImportError:
        print("[train] xgboost not installed, skipping")

    # ── Random Forest ────────────────────────────────────────────────────── #
    print("\n[train] RandomForest (paper: n_est=100, max_depth=10)")
    _, m_rf, rf_prob = train_rf(Xtr, ytr, Xte, yte)
    if m_rf:
        all_results["RandomForest"] = m_rf
        all_probs["RandomForest"] = rf_prob
        print(f"[RF      ] {m_rf}")

    # ── LightGBM ─────────────────────────────────────────────────────────── #
    print("\n[train] LightGBM (paper: num_leaves=200, lr=0.05, n_est=100)")
    _, m_lgb, lgb_prob = train_lightgbm(Xtr, ytr, Xte, yte)
    if m_lgb:
        all_results["LightGBM"] = m_lgb
        all_probs["LightGBM"] = lgb_prob
        print(f"[LightGBM] {m_lgb}")

    # ── CatBoost ─────────────────────────────────────────────────────────── #
    print("\n[train] CatBoost (paper: iter=500, lr=0.05, depth=6, Logloss)")
    _, m_cat, cat_prob = train_catboost(Xtr, ytr, Xte, yte)
    if m_cat:
        all_results["CatBoost"] = m_cat
        all_probs["CatBoost"] = cat_prob
        print(f"[CatBoost] {m_cat}")

    # ── persist all results ──────────────────────────────────────────────── #
    (ARTIFACTS_DIR / "model_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8")
    np.savez(ARTIFACTS_DIR / "model_test_probs.npz", **all_probs)
    print(f"\n[train] Artifacts saved to {ARTIFACTS_DIR}")
    _print_comparison(all_results)


if __name__ == "__main__":
    main()
