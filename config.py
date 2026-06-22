"""Central configuration for the Wildfire IoT Early Warning System.

Single source of truth for the top-10 features, risk thresholds, node grid,
storage paths and the timing contract from the architecture diagram.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
for _d in (DATA_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Top-10 ERA5 features (Table 9 of the paper, kept in rank order) ---------
TOP_10_FEATURES = [
    "swvl1",   # surface soil moisture (layer 1)
    "mn2t",    # minimum 2 m air temperature
    "lgws",    # large-scale wind speed
    "pev",     # potential evapotranspiration
    "DOY",     # day of year (seasonality)
    "gwd",     # wind direction
    "blh",     # boundary layer height
    "mgws",    # medium-scale / mean-gust wind speed
    "vilwd",   # divergence of wind
    "swvl2",   # soil moisture (layer 2)
]
N_FEATURES = len(TOP_10_FEATURES)
TARGET = "fire"  # binary: 1 = fire, 0 = no fire

# --- Sequence / model shape --------------------------------------------------
SEQ_LEN = 7            # 7-day weather window feeding the LSTM
TRAIN_TEST_SPLIT = 0.70

# --- IoT virtual environment -------------------------------------------------
N_NODES = 100          # 10 x 10 grid of virtual sensors over BC
GRID = (10, 10)
# Bounding box for British Columbia (approx) used to place virtual nodes
BC_BOUNDS = dict(lat_min=48.3, lat_max=60.0, lon_min=-139.0, lon_max=-114.0)
GRAPH_KNN = 4          # each node connects to its k nearest neighbours

# --- Risk tiers (legend from the architecture diagram) -----------------------
# probability -> (label, hex colour)
RISK_TIERS = [
    (0.05, "Low",      "#2dd4bf"),
    (0.15, "Medium",   "#f59e0b"),
    (0.30, "High",     "#f97316"),
    (1.01, "Critical", "#ef4444"),
]

def risk_tier(p: float):
    for thresh, label, colour in RISK_TIERS:
        if p < thresh:
            return label, colour
    return RISK_TIERS[-1][1], RISK_TIERS[-1][2]

# --- Timing contract ---------------------------------------------------------
XGB_ALERT_EVERY_S = 60        # fast tabular alert cadence
GATLSTM_FORECAST_EVERY_S = 7 * 60   # 7-minute GAT-LSTM forecast cadence
MAX_E2E_LATENCY_S = 60

# --- Infrastructure endpoints (overridable via env in docker-compose) --------
import os
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "sensor.readings")
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "dev-token")
INFLUX_ORG = os.getenv("INFLUX_ORG", "wildfire")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "sensors")
SQLITE_PATH = os.getenv("SQLITE_PATH", str(ARTIFACTS_DIR / "ews.db"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

MODEL_PATH = ARTIFACTS_DIR / "gat_lstm.pt"
SCALER_PATH = ARTIFACTS_DIR / "scaler.json"
XGB_PATH = ARTIFACTS_DIR / "xgb.json"
GRAPH_PATH = ARTIFACTS_DIR / "graph.json"
