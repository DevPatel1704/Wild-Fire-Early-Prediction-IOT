"""Layer 7 API (LIVE box). FastAPI serves node topology + alert history over
REST and fans out live predictions over WebSocket. A background thread consumes
the `predictions` Kafka topic and pushes each message to all connected clients.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import (KAFKA_BOOTSTRAP, N_NODES, RISK_TIERS, XGB_ALERT_EVERY_S,
                    GATLSTM_FORECAST_EVERY_S, MAX_E2E_LATENCY_S, ARTIFACTS_DIR)
from api.db import init_db, recent_alerts, save_predictions
from iot.graph import load_graph

PRED_TOPIC = "predictions"
_clients: set[WebSocket] = set()
_loop: asyncio.AbstractEventLoop | None = None


def _kafka_loop():
    """Blocking Kafka consumer -> schedule broadcast on the asyncio loop."""
    try:
        from kafka import KafkaConsumer
    except Exception as e:
        print(f"[api] kafka consumer disabled: {e}")
        return
    consumer = KafkaConsumer(
        PRED_TOPIC, bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="latest", group_id="api-fanout",
    )
    for msg in consumer:
        if _loop:
            asyncio.run_coroutine_threadsafe(_broadcast(msg.value), _loop)


async def _broadcast(payload):
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    init_db()
    # ensure eval directory exists so StaticFiles mount succeeds
    (ARTIFACTS_DIR / "eval").mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_kafka_loop, daemon=True).start()
    yield


app = FastAPI(title="Wildfire IoT Early Warning System", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# serve evaluation report + PNG images at /eval/*
_eval_dir = ARTIFACTS_DIR / "eval"
_eval_dir.mkdir(parents=True, exist_ok=True)
app.mount("/eval", StaticFiles(directory=str(_eval_dir)), name="eval-static")


@app.get("/health")
def health():
    return {"status": "ok", "nodes": N_NODES,
            "xgb_every_s": XGB_ALERT_EVERY_S,
            "gatlstm_every_s": GATLSTM_FORECAST_EVERY_S,
            "max_latency_s": MAX_E2E_LATENCY_S}


@app.get("/nodes")
def nodes():
    coords, adj = load_graph()
    edges = [[i, j] for i in range(len(adj)) for j in range(len(adj))
             if adj[i][j] and i < j]
    tiers = [{"label": l, "colour": c, "max": t} for t, l, c in RISK_TIERS]
    return {"coords": coords, "edges": edges, "tiers": tiers}


@app.get("/alerts")
def alerts(limit: int = 50):
    return {"alerts": recent_alerts(limit=limit)}


@app.get("/eval-results")
def eval_results():
    """Return per-model metric comparison for the dashboard Results page."""
    rp = ARTIFACTS_DIR / "model_results.json"
    if not rp.exists():
        return {"results": []}
    raw = json.loads(rp.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        results = [{"model": k, **v} for k, v in raw.items()]
    else:
        results = raw
    return {"results": results}


@app.post("/demo/push")
async def demo_push(request: Request):
    """Accept a prediction payload and broadcast it to all WebSocket clients.

    Used by demo_fire.py to push fire scenarios without Kafka.
    """
    payload = await request.json()
    await _broadcast(payload)
    rows = [
        (payload["ts"], p["node"], p.get("model", "demo"), p["prob"], p["tier"])
        for p in payload.get("preds", [])
    ]
    if rows:
        save_predictions(rows)
    return {"ok": True, "n": len(rows)}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()   # keepalive / ignore client msgs
    except WebSocketDisconnect:
        _clients.discard(websocket)
