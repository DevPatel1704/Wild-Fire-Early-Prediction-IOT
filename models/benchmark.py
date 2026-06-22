"""Layer 5 (offline boxes): evaluation-only benchmark for Random Forest and
CatBoost, so you can reproduce the paper-style comparison table alongside the
LIVE GAT-LSTM / XGBoost models. These models are NOT served; they exist for
the offline comparison the architecture diagram marks 'evaluation only'.

Run:  python -m models.benchmark
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models.train import metrics
from pipeline.sequences import prepare
from data.build_dataset import build
from iot.graph import save_graph


def flatten(X, y):
    return X[:, -1].reshape(-1, X.shape[-1]), y.reshape(-1)


def main():
    build(); save_graph()
    (Xtr, ytr), (Xte, yte) = prepare()
    xtr, ytr_f = flatten(Xtr, ytr)
    xte, yte_f = flatten(Xte, yte)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                min_samples_split=2, class_weight="balanced", n_jobs=-1)
    rf.fit(xtr, ytr_f)
    print(f"[RandomForest] {metrics(yte_f, rf.predict_proba(xte)[:, 1])}")

    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                loss_function="Logloss", verbose=False,
                                auto_class_weights="Balanced")
        cb.fit(xtr, ytr_f)
        print(f"[CatBoost    ] {metrics(yte_f, cb.predict_proba(xte)[:, 1])}")
    except ImportError:
        print("[CatBoost    ] not installed (pip install catboost), skipped")


if __name__ == "__main__":
    main()
