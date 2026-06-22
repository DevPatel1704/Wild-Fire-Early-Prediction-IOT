"""Layer 5 inference. Wraps the trained models so the stream processor can ask
for a fast XGBoost alert (per latest frame) or a GAT-LSTM forecast (per window).
Handles scaling and tiering. Degrades gracefully if a model file is missing.
"""
from __future__ import annotations
import numpy as np
import torch
from config import (N_FEATURES, MODEL_PATH, XGB_PATH, risk_tier)
from models.gat_lstm import GATLSTM
from iot.graph import load_graph
from pipeline.sequences import apply_scaler


class RiskEngine:
    def __init__(self, device="cpu"):
        self.device = device
        self.coords, adj = load_graph()
        self.adj = torch.tensor(adj, dtype=torch.float32, device=device)

        self.gat = GATLSTM(N_FEATURES).to(device)
        if MODEL_PATH.exists():
            self.gat.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            self.gat.eval()
            self.gat_ready = True
        else:
            self.gat_ready = False
            print("[infer] GAT-LSTM weights missing, run models.train first")

        self.xgb = None
        if XGB_PATH.exists():
            try:
                import xgboost as xgb
                self.xgb = xgb.XGBClassifier()
                self.xgb.load_model(str(XGB_PATH))
            except Exception as e:
                print(f"[infer] xgboost unavailable: {e}")

    def _scale(self, arr):
        return apply_scaler(np.asarray(arr, dtype=np.float32))

    def xgb_alert(self, frame):
        """frame: [N, F] latest reading per node -> per-node prob (60s cadence)."""
        if self.xgb is None:
            return None
        prob = self.xgb.predict_proba(self._scale(frame))[:, 1]
        return self._tiered(prob, "xgboost")

    def gatlstm_forecast(self, window):
        """window: [T, N, F] -> per-node prob (7-min cadence). Main model."""
        if not self.gat_ready:
            return None
        x = torch.tensor(self._scale(window)[None], device=self.device)  # [1,T,N,F]
        with torch.no_grad():
            prob = self.gat.predict_proba(x, self.adj).cpu().numpy()[0]
        return self._tiered(prob, "gat-lstm")

    def _tiered(self, prob, model):
        out = []
        for node, p in enumerate(prob):
            label, colour = risk_tier(float(p))
            out.append({"node": node, "lat": self.coords[node][0],
                        "lon": self.coords[node][1], "prob": float(p),
                        "tier": label, "colour": colour, "model": model})
        return out
