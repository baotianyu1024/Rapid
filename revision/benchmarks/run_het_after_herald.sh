#!/usr/bin/env bash
set -euo pipefail

herald_pid="${RAPID_HERALD_QUEUE_PID:?set RAPID_HERALD_QUEUE_PID}"
while kill -0 "$herald_pid" 2>/dev/null; do
  sleep 30
done

cd "$(dirname "$0")/.."
HET_STAGE_ROOT=/dev/shm/rapid_het_holdout_d32_main_20261104 \
  HET_DATASETS=kaggle HET_MODELS=deepfm \
  HET_TAG=sgd_510_stable_bce HET_LR=0.01 \
  bash code/run_het_d32_holdout.sh
python3 code/summarize_het_stable_deepfm.py
