#!/usr/bin/env bash
# Local (non-docker) launcher. Assumes Kafka + InfluxDB already running and
# the model already trained. Starts API, stream processor and simulator.
set -e
cd "$(dirname "$0")/.."

echo "==> training (skip if artifacts/gat_lstm.pt exists)"
[ -f artifacts/gat_lstm.pt ] || python -m models.train --epochs 20

echo "==> starting API on :8000"
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API=$!

sleep 3
echo "==> starting stream processor"
python -m stream.processor &
STREAM=$!

echo "==> starting 100-node IoT simulator"
python -m iot.simulator --tick 1.0 --hot 6 &
SIM=$!

echo "==> dashboard: cd dashboard && npm install && npm run dev"
trap "kill $API $STREAM $SIM" EXIT
wait
