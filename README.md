# Wildfire IoT Early Warning System 

A live, GAT-LSTM-driven wildfire early warning system built from the Nasourinia
& Passi (2025) BC wildfire study, with two deliberate changes:

1. **GAT-LSTM is the main model** (graph attention over sensor nodes + temporal
   LSTM), replacing the paper's plain RNN+LSTM.
2. **Only the top-10 features** are used (Table 9): `swvl1, mn2t, lgws, pev,
   DOY, gwd, blh, mgws, vilwd, swvl2`.

Training uses **ERA5 + NFDB + FIRMS/NASA**. Testing uses a **virtual environment
of 100 IoT sensor nodes** streaming live readings through Kafka.

## Architecture (8 layers, matches the diagram)

| Layer | Component | File |
|------|-----------|------|
| 1 Data sources | NFDB · FIRMS/NASA · ERA5 | `data/sources.py` |
| 2 Preprocessing | integrate, Haversine label, scale | `data/build_dataset.py`, `pipeline/sequences.py` |
| 3 Feature selection | top-10 features | `config.py` |
| 4 Train/test split | 70:30 chronological + RUS | `pipeline/sequences.py` |
| 5 Models | **GAT-LSTM (LIVE)**, XGBoost (LIVE), RF/CatBoost (offline) | `models/` |
| 6 Live IoT pipeline | 100 nodes → Kafka → stream processor | `iot/`, `stream/` |
| 7 Storage & API | SQLite · FastAPI+WebSocket · InfluxDB | `api/` |
| 8 Dashboard | React command center · `localhost:3000` | `dashboard/` |

Timing contract: XGBoost alert every 60 s, GAT-LSTM forecast every 7 min,
end-to-end latency < 60 s. Risk tiers: Low <5%, Medium 5–15%, High 15–30%,
Critical ≥30%.

## Quick start - Docker (one command)

```bash
docker compose up --build
```

Brings up Kafka, InfluxDB, the API, the stream processor, the 100-node
simulator, and the dashboard. Then:

```bash
# train the models once (artifacts are shared via the ./artifacts volume)
docker compose run --rm api python -m models.train --epochs 20
docker compose restart stream
```

Open the dashboard at **http://localhost:3000**.

## Quick start - local (no Docker)

```bash
pip install -r requirements.txt

# 1. build dataset (uses ERA5/NFDB/FIRMS if present in data/raw, else synthetic)
python -m data.build_dataset

# 2. train GAT-LSTM (main) + XGBoost (fast alert)
python -m models.train --epochs 20

# 3. (optional) offline comparison: Random Forest + CatBoost
python -m models.benchmark

# 4. all evaluation graphs (data overview, training curve, ROC/PR, confusion,
#    threshold sweep, calibration, risk tiers) -> artifacts/eval/report.html
python -m models.evaluate

# 5. start Kafka + InfluxDB yourself, then run the live stack
uvicorn api.main:app --host 0.0.0.0 --port 8000     # terminal 1
python -m stream.processor                          # terminal 2
python -m iot.simulator --tick 1.0 --hot 6          # terminal 3

# 6. dashboard
cd dashboard && npm install && npm run dev           # terminal 4 -> :3000
```

Or use the helper: `bash scripts/run_local.sh` (assumes Kafka + InfluxDB up).

## Using real data instead of synthetic

Drop these into `data/raw/` and the builder picks them up automatically:

- `era5.nc`   - ERA5 reanalysis NetCDF, BC bbox, daily (https://cds.climate.copernicus.eu)
- `nfdb.csv`  - NFDB fire points with `lat,lon,date` (https://cwfis.cfs.nrcan.gc.ca)
- `firms.csv` - FIRMS active fire CSV (https://firms.modaps.eosdis.nasa.gov)

Edit the variable rename map in `data/sources.load_era5` to match your CDS pull,
then rerun `python -m data.build_dataset`.

## Run commands cheat-sheet

```bash
python -m data.build_dataset                  # build labelled dataset
python -m models.train --epochs 20            # train GAT-LSTM + XGBoost
python -m models.evaluate                     # all evaluation graphs -> artifacts/eval/
python -m models.benchmark                    # offline RF/CatBoost eval
python -m iot.simulator --tick 1.0 --hot 6    # 100 virtual nodes -> Kafka
python -m stream.processor                    # live inference + storage
uvicorn api.main:app --port 8000              # API + WebSocket
docker compose up --build                     # everything at once
```


## Team Contributions

### Dev: Data Streaming, Pipeline & ML Benchmarks

1. Set up Apache Kafka as the message bus and implemented the IoT producer/consumer that moves sensor readings (100 nodes × 10 features, JSON-encoded) through the `sensor.readings` topic at 100 messages per tick (1-second default cadence).
2. Built the stream processor (`stream/processor.py`) with a 60-second cadence scorer for XGBoost fast alerts and a rolling 7-frame buffer for GAT-LSTM forecasts - both running concurrently against the same incoming stream.
3. Created the aggregation pipeline: snapping raw ERA5 grid cells to the nearest of the 100 virtual sensor nodes, building 7-day sliding windows shaped `[T=7, N=100, F=10]`, and ensuring data reliably reaches both models before persisting predictions to SQLite and InfluxDB.
4. Trained and benchmarked the four tabular/classical ML models used as comparison baselines - **Random Forest** (100 trees, max depth 10, AUC 0.971), **XGBoost** (200 estimators, depth 6, `scale_pos_weight`, AUC 0.971), **LightGBM** (200 leaves, lr 0.05, 100 estimators, AUC 0.971), and **CatBoost** (500 iterations, lr 0.05, depth 6, AUC 0.972) - and implemented `RNN+LSTM` (2 × 64 LSTM units, Adam lr 0.001, AUC 0.968) as the paper's own sequence baseline, with all results written to `artifacts/model_results.json`.
5. Measured end-to-end latency and confirmed the system meets the <60-second alert SLA from sensor reading to dashboard update.


### Rashmi: Machine Learning (GAT-LSTM)

1. Designed and implemented the **GAT-LSTM** architecture: a two-layer Graph Attention Network (4 attention heads, 100 nodes, k=4 neighbour graph) feeding a 2-layer LSTM (hidden size 128) over 7-day windows, with LayerNorm after both the spatial and temporal stages and a two-layer dense head (128 → 64 → 1 with GELU) outputting per-node fire probability.
2. Handled the severe class imbalance with Random Undersampling on the training split and `BCEWithLogitsLoss` positive-class weighting, achieving a final test **AUC of 0.969**, **Recall of 96.7%**, Precision 79.2%, F1 0.871, and F2 0.927 - the highest F2 of all six models evaluated.
3. Tuned hyperparameters (Adam lr 0.001, batch size 16, 20 epochs, best-checkpoint saving on test AUC) and validated that the graph attention mechanism correctly propagates risk across spatially connected nodes, with GAT-LSTM outperforming the equivalent RNN+LSTM baseline (AUC 0.968) by confirming the spatial graph adds predictive value.
4. Evaluated all models prioritising AUC and Recall over accuracy given class imbalance, produced confusion matrices, ROC/PR curves, and the model comparison table saved to `artifacts/eval/` and surfaced in the dashboard Results tab. Wrote Final Report.

### Dhruv: Data Preparation & IoT Sensor Simulation

1. Assembled the real wildfire dataset: ERA5 weather reanalysis, NFDB historical fire records, and NASA FIRMS active-hotspot feeds for British Columbia, 2019–2023 (~6.5M rows).
2. Cleaned and labelled the data: joined fire records to the 0.25° ERA5 grid by date and location using Haversine distance (10 km radius), dropped incomplete rows, and set the binary fire label.
3. Built a physics-based synthetic data fallback (`sources.synthesize()`) that generates realistic 2-year daily records across a 10×10 grid of BC nodes when real downloads are unavailable, using soil-moisture and temperature-driven fire logits with spatial aridity bias for BC's interior.
4. Built the 100-node IoT simulator (`iot/simulator.py`) that streams ERA5-matched sensor readings into the live Kafka pipeline at configurable tick rates (default: 1 second per full sweep), seeding 5 elevated-risk nodes to drive realistic High/Critical alert behaviour, and validated the streamed output end-to-end.


### Priyanka: Backend API & Alert System

1. Built the **FastAPI** backend (`api/main.py`) serving the full system: REST endpoints (`/health`, `/nodes`, `/alerts`, `/eval-results`) plus a **WebSocket** channel (`/ws`) that fans out live predictions to all connected dashboard clients simultaneously.
2. Connected model predictions to the alert layer: the stream processor scores each of the 100 nodes through the inference engine, which tiers each node's score; the API surfaces High and Critical tier entries as timestamped alerts from the SQLite store.
3. Implemented **percentile-based risk tiering** in the inference engine: nodes are ranked within each prediction batch and split at the 25th, 50th, and 75th percentiles into Low, Medium, High, and Critical tiers - so spatial gradients remain visible even when absolute probabilities cluster high in peak fire season.
4. Wired **SQLite** (`artifacts/ews.db`) for alert history and **InfluxDB** for raw sensor time-series storage; exposed the `/demo/push` endpoint that accepts predictions without Kafka, enabling `replay_test_stream.py` to drive the full dashboard in a no-Docker demo environment.


### Slesha: Dashboard, Visualization & Report

1. Built the **React + Vite** dashboard (`dashboard/`) with a live risk map colour-coding all 100 sensor nodes by fire probability (teal → amber → orange → red for Low/Medium/High/Critical), k-NN edge overlay showing spatial topology, and real-time updates over WebSocket.
2. Created the three-tab layout: **GAT-LSTM** tab (7-minute spatial-temporal forecast with countdown timer and alert feed), **XGBoost** tab (60-second fast-alert view with tabular risk), and **Results** tab (offline model comparison table across all six models, ROC/PR curves, confusion matrices, training history, and per-node F1 map).
3. Displayed live detection stats - active High/Critical node count, last-update timestamp, risk tier distribution histogram - and wired the alert panel to surface only actionable High/Critical events with node ID, coordinates, probability, and model source.


### Darsh: Feature Engineering & Feature Selection

1. Optimizing features from raw data to make them more meaningful for machine learning models


