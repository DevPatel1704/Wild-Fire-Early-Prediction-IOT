import React from "react";

const TIER_META = [
  ["Low", "#2dd4bf"],
  ["Medium", "#f59e0b"],
  ["High", "#f97316"],
  ["Critical", "#ef4444"],
];

export function RiskBars({ risk, total }) {
  const counts = { Low: 0, Medium: 0, High: 0, Critical: 0 };
  Object.values(risk).forEach((r) => { if (r?.tier) counts[r.tier]++; });
  const n = total || 1;
  return (
    <div className="bars">
      {TIER_META.map(([label, colour]) => (
        <div className="bar-row" key={label}>
          <span className="lbl">{label}</span>
          <span className="bar-track">
            <span className="bar-fill" style={{ width: `${(counts[label] / n) * 100}%`, background: colour }} />
          </span>
          <span className="n">{counts[label]}</span>
        </div>
      ))}
    </div>
  );
}

export function AlertFeed({ alerts, tierColour }) {
  if (!alerts.length) return <div className="empty">No High or Critical alerts. Monitoring.</div>;
  return (
    <div className="feed">
      {alerts.map((a, i) => (
        <div className="alert" key={i}>
          <span className="mark" style={{ background: tierColour(a.prob) }} />
          <span className="who">
            node {a.node} · <small>{a.model}</small>
            <br />
            <small>{new Date(a.ts).toLocaleTimeString()}</small>
          </span>
          <span className="prob" style={{ color: tierColour(a.prob) }}>
            {(a.prob * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  );
}
