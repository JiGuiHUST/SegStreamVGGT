#!/usr/bin/env bash
# Launch the interactive demo server.
#   ./run.sh                 # 0.0.0.0:8787, auto-select GPU per job
#   DEMO_GPU=2 ./run.sh      # pin physical GPU 2 (debug)
#   DEMO_PORT=9000 ./run.sh  # another port
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${DEMO_PYTHON:-/data/luhangze/miniconda3/envs/segvggt/bin/python}"
GPU="${DEMO_GPU:-auto}"
HOST="${DEMO_HOST:-0.0.0.0}"
PORT="${DEMO_PORT:-8787}"

if [[ ! -x "$PYTHON" ]]; then
  echo "python not found: $PYTHON (set DEMO_PYTHON)" >&2
  exit 1
fi

mkdir -p logs
exec "$PYTHON" server.py --host "$HOST" --port "$PORT" --gpu "$GPU" "$@"
