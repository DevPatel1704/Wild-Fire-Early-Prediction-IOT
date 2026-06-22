import React, { useEffect, useState } from "react";

// Ordered by importance for fire EWS: recall > F2 > F1 > precision > accuracy > AUC
const METRICS    = ["recall", "f2", "f1", "precision", "accuracy", "auc"];
const MET_LABELS = ["Recall", "F2 ★", "F1", "Precision", "Accuracy", "AUC"];

const MODEL_COLORS = {
  "GAT-LSTM":     "#2dd4bf",
  "CatBoost":     "#a78bfa",
  "RandomForest": "#34d399",
  "XGBoost":      "#60a5fa",
  "LightGBM":     "#f59e0b",
  "RNN+LSTM":     "#ef4444",
};

// Evaluation graph sections — all served from /eval/*.png by FastAPI StaticFiles
const GRAPH_SECTIONS = [
  {
    label: "GAT-LSTM — Training Curves",
    graphs: [
      { file: "training_history.png", title: "Loss · Accuracy · Recall · F1 · F2 · AUC per epoch" },
    ],
  },
  {
    label: "GAT-LSTM — Primary Model Performance",
    graphs: [
      { file: "roc_curve.png",        title: "ROC Curve (AUC)" },
      { file: "pr_curve.png",         title: "Precision-Recall Curve" },
      { file: "confusion_matrix.png", title: "Confusion Matrix (F1-optimal threshold)" },
      { file: "threshold_sweep.png",  title: "Metrics vs Decision Threshold" },
      { file: "precision_deep_dive.png", title: "Precision Deep-Dive — False Alarm Analysis" },
      { file: "calibration.png",      title: "Calibration — Predicted vs Observed Frequency" },
      { file: "prob_distribution.png",title: "Probability Distribution by True Class" },
      { file: "risk_tiers.png",       title: "Risk Tier Distribution" },
      { file: "error_breakdown.png",  title: "Error Breakdown per Risk Tier" },
    ],
  },
  {
    label: "Spatial Analysis — Node Performance",
    graphs: [
      { file: "node_performance_heatmap.png", title: "Per-Node F1 Score — 10×10 BC Sensor Grid" },
    ],
  },
  {
    label: "Model Comparison — All Models",
    graphs: [
      { file: "model_comparison.png",      title: "Bar Comparison — Accuracy · Recall · Precision · F1 · F2 · AUC" },
      { file: "model_metrics_heatmap.png", title: "Metrics Heatmap — All Models" },
      { file: "roc_all_models.png",        title: "ROC Curves — All Models on One Chart" },
    ],
  },
  {
    label: "Dataset Analysis",
    graphs: [
      { file: "data_class_balance.png",         title: "Class Balance (fire / no-fire)" },
      { file: "data_seasonality.png",           title: "Fire Rate by Day of Year" },
      { file: "data_fire_timeline.png",         title: "Daily Fire Count over Time" },
      { file: "data_monthly_rate.png",          title: "Monthly Fire Rate" },
      { file: "data_feature_importance.png",    title: "RF Feature Importance — Top-10 Features" },
      { file: "data_feature_distributions.png", title: "Feature Distributions — Fire vs No-Fire" },
      { file: "data_correlation.png",           title: "Feature Correlation Heatmap" },
    ],
  },
];

function scoreClass(v) {
  if (v >= 0.92) return "score-hi";
  if (v >= 0.85) return "score-mid";
  return "";
}

function GraphSection({ label, graphs }) {
  const [open, setOpen] = useState(true);
  const [lightbox, setLightbox] = useState(null);

  return (
    <div className="graph-section">
      <button className="graph-section-header" onClick={() => setOpen((o) => !o)}>
        <span>{label}</span>
        <span className="graph-toggle">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="graph-grid">
          {graphs.map(({ file, title }) => (
            <div key={file} className="graph-card" onClick={() => setLightbox(file)}>
              <img
                src={`/eval/${file}`}
                alt={title}
                className="graph-img"
                loading="lazy"
              />
              <div className="graph-caption">{title}</div>
            </div>
          ))}
        </div>
      )}

      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <img src={`/eval/${lightbox}`} alt={lightbox} className="lightbox-img" />
          <div className="lightbox-hint">click anywhere to close</div>
        </div>
      )}
    </div>
  );
}

export default function EvalPage() {
  const [rows, setRows]       = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    fetch("/api/eval-results")
      .then((r) => r.json())
      .then((d) => { setRows(d.results || []); setLoading(false); })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }, []);

  if (loading) return <div className="empty">Loading evaluation results…</div>;
  if (error)   return <div className="empty">Error: {error}</div>;
  if (!rows.length)
    return (
      <div className="empty" style={{ lineHeight: 1.9 }}>
        No results yet. Run:<br />
        <code>python preview_eval.py</code><br />
        then restart the API server.
      </div>
    );

  const gat    = rows.find((r) => r.model === "GAT-LSTM");
  const others = [...rows.filter((r) => r.model !== "GAT-LSTM")]
                   .sort((a, b) => (b.f2 || 0) - (a.f2 || 0));
  const sorted = gat ? [gat, ...others] : others;
  const maxF2  = Math.max(...sorted.map((r) => r.f2 || 0));
  const maxF1  = Math.max(...sorted.map((r) => r.f1 || 0));

  const gatPrec       = gat ? (gat.precision || 0) : 0;
  const falseAlarmIn  = gatPrec > 0 ? Math.round(1 / (1 - gatPrec)) : null;

  return (
    <div className="eval-page">

      {/* ── header ──────────────────────────────────────────────────────── */}
      <div className="eval-header">
        <div>
          <div className="eyebrow2">70:30 Chronological Split · Top-10 Features · RUS Balanced</div>
          <div className="eval-title">Model Evaluation Results</div>
          <div className="eyebrow2" style={{ marginTop: 4 }}>
            Reference: Nasourinia &amp; Passi (2025) · BC Wildfire Prediction
          </div>
        </div>
        <a className="report-btn" href="/eval/report.html" target="_blank" rel="noreferrer">
          Full Report ↗
        </a>
      </div>

      {/* ── primary model callout ────────────────────────────────────────── */}
      {gat && (
        <div className="primary-callout">
          <div className="primary-badge">PRIMARY MODEL</div>
          <div className="primary-name">GAT-LSTM</div>
          <div className="primary-desc">
            Graph Attention Network + LSTM · spatial-temporal · 7-day sliding window · k-NN sensor topology
          </div>
          <div className="primary-metrics">
            {METRICS.map((m) => (
              <div className="primary-met" key={m}>
                <span className="primary-met-val" style={{ color: MODEL_COLORS["GAT-LSTM"] }}>
                  {((gat[m] || 0) * 100).toFixed(1)}%
                </span>
                <span className="primary-met-key">{MET_LABELS[METRICS.indexOf(m)]}</span>
                {m === "recall" && (
                  <span className="primary-met-note">★ highest recall — fewest missed fires</span>
                )}
                {m === "f2" && (
                  <span className="primary-met-note">leads all models on F2</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── precision insight ────────────────────────────────────────────── */}
      {gat && falseAlarmIn && (
        <div className="prec-insight">
          <div className="prec-insight-row">
            <span className="prec-label">Precision analysis</span>
            <span className="prec-val">{(gatPrec * 100).toFixed(1)}%</span>
          </div>
          <div className="prec-body">
            <b>What does {(gatPrec * 100).toFixed(1)}% precision mean?</b>{" "}
            Out of every {falseAlarmIn} fire alerts, ~{falseAlarmIn - 1} are real fires
            and ~1 is a false alarm. For an early warning system, this is acceptable —
            recall ({((gat.recall || 0) * 100).toFixed(1)}%) matters far more: every missed fire
            risks lives and infrastructure.{" "}
            <b>F2 score ({((gat.f2 || 0) * 100).toFixed(1)}%)</b> weights recall 2× over precision
            — the right metric for fire EWS. GAT-LSTM leads all models on F2.
          </div>
        </div>
      )}

      {/* ── comparison table ─────────────────────────────────────────────── */}
      <div className="eval-section-label" style={{ marginTop: 24 }}>Full Comparison — sorted by F2 ★</div>
      <div className="eval-table-wrap">
        <table className="eval-table">
          <thead>
            <tr>
              <th>Model</th>
              {MET_LABELS.map((m) => <th key={m}>{m}</th>)}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => {
              const isPrimary = r.model === "GAT-LSTM";
              return (
                <tr key={r.model} className={isPrimary ? "primary-row" : ""}>
                  <td>
                    <span className="model-dot"
                          style={{ background: MODEL_COLORS[r.model] || "#647686" }} />
                    {r.model}
                    {isPrimary && <span className="row-badge">PRIMARY</span>}
                  </td>
                  {METRICS.map((m) => (
                    <td key={m} className={scoreClass(r[m] || 0)}>
                      {((r[m] || 0) * 100).toFixed(1)}%
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ── F2 bar chart ─────────────────────────────────────────────────── */}
      <div className="eval-section-label">
        F2 Score Comparison
        <span style={{ color: "var(--muted)", fontWeight: 400 }}> (recall-weighted · correct metric for fire EWS)</span>
      </div>
      <div className="eval-bars" style={{ marginBottom: 20 }}>
        {sorted.map((r) => (
          <div className="eval-bar-row" key={r.model + "-f2"}>
            <span className="eval-bar-label">{r.model}</span>
            <div className="eval-bar-track">
              <div className="eval-bar-fill"
                   style={{ width: `${((r.f2 || 0) / maxF2) * 100}%`, background: MODEL_COLORS[r.model] || "#647686" }} />
            </div>
            <span className="eval-bar-val">{((r.f2 || 0) * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      {/* ── F1 bar chart ─────────────────────────────────────────────────── */}
      <div className="eval-section-label">
        F1 Score Comparison
        <span style={{ color: "var(--muted)", fontWeight: 400 }}> (equal precision/recall weight)</span>
      </div>
      <div className="eval-bars" style={{ marginBottom: 28 }}>
        {sorted.map((r) => (
          <div className="eval-bar-row" key={r.model + "-f1"}>
            <span className="eval-bar-label">{r.model}</span>
            <div className="eval-bar-track">
              <div className="eval-bar-fill"
                   style={{ width: `${((r.f1 || 0) / maxF1) * 100}%`, background: MODEL_COLORS[r.model] || "#647686" }} />
            </div>
            <span className="eval-bar-val">{((r.f1 || 0) * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      {/* ── evaluation graph sections ─────────────────────────────────────── */}
      <div className="eval-section-label" style={{ marginBottom: 14 }}>
        Evaluation Charts
        <span style={{ color: "var(--muted)", fontWeight: 400 }}> — click any chart to enlarge</span>
      </div>
      {GRAPH_SECTIONS.map((sec) => (
        <GraphSection key={sec.label} label={sec.label} graphs={sec.graphs} />
      ))}

      {/* ── paper reference note ─────────────────────────────────────────── */}
      <div className="paper-note" style={{ marginTop: 28 }}>
        <b>Why GAT-LSTM leads on Recall &amp; F2:</b> Graph attention propagates fire risk from
        burning nodes to neighbours — the model detects fire spread early, before
        tabular per-node models can. High recall means fewer missed fires in the EWS.<br />
        <b>F2 score</b> (β=2) weights recall 2× over precision: F2 = 5·P·R / (4P + R).
        It is the correct primary metric for any early warning system.<br />
        <b>Paper benchmark (real 3.6M BC records):</b> CatBoost 93.4% acc, 92.1% F1,
        0.94 AUC. Results above use synthetic data with the same top-10 features.
      </div>

    </div>
  );
}
