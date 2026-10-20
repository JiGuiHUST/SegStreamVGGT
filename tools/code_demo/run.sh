#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${DEMO_PYTHON:-/data/luhangze/miniconda3/envs/modelscope/bin/python}"
exec "$PYTHON" render_video.py "$@"
