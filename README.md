# Wildfire IoT Early Warning System (British Columbia)

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

## Quick start — Docker (one command)

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

## Quick start — local (no Docker)

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

- `era5.nc`   — ERA5 reanalysis NetCDF, BC bbox, daily (https://cds.climate.copernicus.eu)
- `nfdb.csv`  — NFDB fire points with `lat,lon,date` (https://cwfis.cfs.nrcan.gc.ca)
- `firms.csv` — FIRMS active fire CSV (https://firms.modaps.eosdis.nasa.gov)

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
