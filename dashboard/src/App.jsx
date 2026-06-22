import React, { useEffect, useMemo, useRef, useState } from "react";
import NodeMap from "./components/NodeMap.jsx";
import { RiskBars, AlertFeed } from "./components/Panels.jsx";
import EvalPage from "./components/EvalPage.jsx";

const TIERS = [
  [0.05, "Low",      "#2dd4bf"],
  [0.15, "Medium",   "#f59e0b"],
  [0.30, "High",     "#f97316"],
  [1.01, "Critical", "#ef4444"],
];
const tierColour = (p) => (TIERS.find(([t]) => p < t) || TIERS[3])[2];

export default function App() {
  const [topo, setTopo]               = useState({ coords: [], edges: [] });
  const [page, setPage]               = useState("gatlstm");   // "gatlstm" | "xgboost" | "results"
  const [gatlstmRisk, setGatlstmRisk] = useState({});
  const [xgbRisk,     setXgbRisk]     = useState({});
  const [gatlstmAlerts, setGatlstmAlerts] = useState([]);
  const [xgbAlerts,     setXgbAlerts]     = useState([]);
  const [gatlstmCadence, setGatlstmCadence] = useState(7 * 60);
  const [xgbCadence,     setXgbCadence]     = useState(60);
  const [live, setLive]               = useState(false);
  const [contract, setContract]       = useState({});
  const wsRef = useRef(null);

  // ── data fetching + WebSocket ──────────────────────────────────────────── //
  useEffect(() => {
    fetch("/api/nodes").then((r) => r.json()).then(setTopo).catch(() => {});
    fetch("/api/health").then((r) => r.json()).then(setContract).catch(() => {});

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    wsRef.current = ws;
    ws.onopen  = () => { setLive(true); ws.send("hi"); };
    ws.onclose = () => setLive(false);

    ws.onmessage = (ev) => {
      const msg   = JSON.parse(ev.data);
      const preds = msg.preds || [];
      const next  = {};
      preds.forEach((p) => { next[p.node] = p; });

      if (msg.kind === "forecast") {
        // GAT-LSTM spatial-temporal forecast
        setGatlstmRisk((prev) => ({ ...prev, ...next }));
        setGatlstmCadence(7 * 60);
        const hi = preds.filter((p) => p.tier === "High" || p.tier === "Critical");
        if (hi.length)
          setGatlstmAlerts((prev) =>
            [...hi.map((p) => ({ ...p, ts: msg.ts })), ...prev].slice(0, 60)
          );
      } else {
        // XGBoost fast alert
        setXgbRisk((prev) => ({ ...prev, ...next }));
        setXgbCadence(60);
        const hi = preds.filter((p) => p.tier === "High" || p.tier === "Critical");
        if (hi.length)
          setXgbAlerts((prev) =>
            [...hi.map((p) => ({ ...p, ts: msg.ts })), ...prev].slice(0, 60)
          );
      }
    };
    return () => ws.close();
  }, []);

  // ── per-model countdown timers ─────────────────────────────────────────── //
  useEffect(() => {
    const t = setInterval(
      () => setGatlstmCadence((c) => (c > 0 ? c - 1 : 7 * 60)), 1000
    );
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    const t = setInterval(
      () => setXgbCadence((c) => (c > 0 ? c - 1 : 60)), 1000
    );
    return () => clearInterval(t);
  }, []);

  // ── active-page derived values ─────────────────────────────────────────── //
  const isGat          = page === "gatlstm";
  const isXgb          = page === "xgboost";
  const isResults      = page === "results";
  const activeRisk     = isGat ? gatlstmRisk     : xgbRisk;
  const activeAlerts   = isGat ? gatlstmAlerts   : xgbAlerts;
  const activeCadence  = isGat ? gatlstmCadence  : xgbCadence;
  const activeMax      = isGat ? 7 * 60          : 60;
  const total          = topo.coords.length || 100;
  const ringPct        = useMemo(
    () => (1 - activeCadence / activeMax) * 100,
    [activeCadence, activeMax]
  );
  const mm = String(Math.floor(activeCadence / 60)).padStart(2, "0");
  const ss = String(activeCadence % 60).padStart(2, "0");

  const bannerClass = isGat ? "gat" : isXgb ? "xgb" : "results";

  return (
    <div className="shell">

      {/* ── top bar ───────────────────────────────────────────────────────── */}
      <header className="topbar">
        <div>
          <div className="eyebrow">British Columbia · IoT Early Warning</div>
          <div className="brand">Wildfire <b>Command Center</b></div>
        </div>

        {/* model page tabs */}
        <div className="model-tabs">
          <button
            className={`tab${isGat ? " active gat" : ""}`}
            onClick={() => setPage("gatlstm")}
          >
            GAT-LSTM
          </button>
          <button
            className={`tab${isXgb ? " active xgb" : ""}`}
            onClick={() => setPage("xgboost")}
          >
            XGBoost
          </button>
          <button
            className={`tab${isResults ? " active results" : ""}`}
            onClick={() => setPage("results")}
          >
            Results
          </button>
        </div>

        <div className="spacer" />
        <div className="stat">
          <span className="k">Nodes</span>
          <span className="v">{total}</span>
        </div>
        <div className="stat">
          <span className="k">Model</span>
          <span className="v">{isGat ? "GAT-LSTM" : isXgb ? "XGBoost" : "Eval"}</span>
        </div>
        <div className="stat">
          <span className="k">Cadence</span>
          <span className="v">{isGat ? "7 min" : isXgb ? "60 s" : "—"}</span>
        </div>
        <div className="live">
          <span className={`dot ${live ? "" : "off"}`} />
          {live ? "LIVE" : "OFFLINE"}
        </div>
      </header>

      {/* ── model description banner ────────────────────────────────────────── */}
      <div className={`page-banner ${bannerClass}`}>
        {isGat
          ? "GAT-LSTM · Graph Attention + LSTM · spatial-temporal 7-day window · main forecast model"
          : isXgb
          ? "XGBoost · Tabular fast-alert model · per-node features · 60-second cadence"
          : "Model Evaluation · all 5 paper models (RF, XGBoost, LightGBM, CatBoost, RNN+LSTM) + GAT-LSTM · 70:30 split"}
      </div>

      {/* ── Results page ───────────────────────────────────────────────────── */}
      {isResults && <EvalPage />}

      {/* ── Live prediction pages ──────────────────────────────────────────── */}
      {!isResults && (
        <div className="grid">
          <section className="panel">
            <h2>Sensor network · {isGat ? "GAT-LSTM" : "XGBoost"} risk</h2>
            <NodeMap
              coords={topo.coords}
              edges={topo.edges}
              risk={activeRisk}
              tierColour={tierColour}
            />
            <div className="legend">
              {TIERS.map(([t, l, c], i) => (
                <span key={i}>
                  <i className="swatch" style={{ background: c }} />
                  {l}&nbsp;{i === 0 ? "< 5%" : i === 3 ? "≥30%" : `< ${Math.round(t * 100)}%`}
                </span>
              ))}
            </div>
          </section>

          <div className="side">
            <section className="panel">
              <h2>Risk distribution</h2>
              <RiskBars risk={activeRisk} total={total} />
            </section>

            <section className="panel">
              <h2>Next {isGat ? "forecast" : "alert"}</h2>
              <div className="cadence">
                <div className="ring" style={{ "--p": ringPct }}>
                  <i>{mm}:{ss}</i>
                </div>
                <div>
                  <div>
                    {isGat ? "GAT-LSTM forecast every " : "XGBoost alert every "}
                    <b>{isGat ? "7 min" : "60 s"}</b>
                  </div>
                  <div style={{ color: "var(--muted)", marginTop: 4 }}>
                    {isGat
                      ? "Graph attention over k-NN sensor topology"
                      : "Fast tabular model, no graph structure"}
                  </div>
                </div>
              </div>
            </section>

            <section className="panel">
              <h2>Alert feed · High / Critical</h2>
              <AlertFeed alerts={activeAlerts} tierColour={tierColour} />
            </section>
          </div>
        </div>
      )}
    </div>
  );
}
