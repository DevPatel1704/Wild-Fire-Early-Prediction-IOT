import React, { useMemo } from "react";

// project lat/lon into the svg box
function project(coords, w, h, pad = 30) {
  const lats = coords.map((c) => c[0]);
  const lons = coords.map((c) => c[1]);
  const [laMin, laMax] = [Math.min(...lats), Math.max(...lats)];
  const [loMin, loMax] = [Math.min(...lons), Math.max(...lons)];
  return coords.map(([la, lo]) => {
    const x = pad + ((lo - loMin) / (loMax - loMin || 1)) * (w - 2 * pad);
    const y = h - pad - ((la - laMin) / (laMax - laMin || 1)) * (h - 2 * pad);
    return [x, y];
  });
}

export default function NodeMap({ coords, edges, risk, tierColour }) {
  const W = 600, H = 480;
  const pts = useMemo(() => project(coords, W, H), [coords]);
  if (!coords.length) return null;

  return (
    <div className="map-wrap">
      <svg viewBox={`0 0 ${W} ${H}`}>
        <defs>
          {["#2dd4bf", "#f59e0b", "#f97316", "#ef4444"].map((c, i) => (
            <radialGradient id={`g${i}`} key={i}>
              <stop offset="0%" stopColor={c} stopOpacity="0.9" />
              <stop offset="100%" stopColor={c} stopOpacity="0" />
            </radialGradient>
          ))}
        </defs>
        {edges.map(([a, b], i) => (
          <line key={i} x1={pts[a][0]} y1={pts[a][1]} x2={pts[b][0]} y2={pts[b][1]}
                stroke="#1d2a35" strokeWidth="0.8" />
        ))}
        {pts.map(([x, y], n) => {
          const p = risk[n]?.prob ?? 0;
          const colour = tierColour(p);
          const r = 4 + p * 9;
          return (
            <g key={n}>
              {p >= 0.15 && <circle cx={x} cy={y} r={r + 12} fill={`url(#g${p >= 0.3 ? 3 : 2})`} />}
              <circle className="node" cx={x} cy={y} r={r} fill={colour}
                      stroke="#0a0e13" strokeWidth="1.2">
                <title>{`node ${n} · ${(p * 100).toFixed(1)}%`}</title>
              </circle>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
