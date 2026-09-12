#!/usr/bin/env bash
# 补实验：传输效率（默认 100 episodes）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/qwh/miniconda3/envs/dqn/bin/python}"
cd "$ROOT"
exec "$PY" supplementary_traffic_experiments/run_transmission_efficiency.py "$@"
