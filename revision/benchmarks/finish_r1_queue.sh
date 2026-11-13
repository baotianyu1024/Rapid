#!/usr/bin/env bash
set -u

queue_pid="${RAPID_HERALD_QUEUE_PID:?set RAPID_HERALD_QUEUE_PID}"
while kill -0 "$queue_pid" 2>/dev/null; do
  sleep 30
done

cd "$(dirname "$0")/.."
for dataset in kaggle terabyte; do
  for model in dlrm dcnv2 wide_deep deepfm; do
    out="/tmp/rapid_herald_r1_20261105/${dataset}_${model}_510steps"
    if [[ -d "$out" ]] && RESULT_DIR="$out" python3 - <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0, 'code')
from summarize_herald_r1 import summarize
summarize(Path(os.environ['RESULT_DIR']))
PY
    then
      echo "VALIDATED $dataset $model $(date -u +%FT%TZ)"
      continue
    fi
    if [[ -d "$out" ]]; then
      mv "$out" "${out}_failed_$(date -u +%Y%m%dT%H%M%S)"
    fi
    echo "RETRY $dataset $model $(date -u +%FT%TZ)"
    RAPID_HERALD_DATASET="$dataset" RAPID_HERALD_MODEL="$model" \
      RAPID_HERALD_STEPS=510 RAPID_HERALD_LOCAL_SHARED=0 \
      timeout -k 15s 7200s bash code/probe_herald_r1.sh \
      > "/tmp/rapid_herald_r1_20261105/${dataset}_${model}_retry_launcher.log" 2>&1
    if RESULT_DIR="$out" python3 - <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0, 'code')
from summarize_herald_r1 import summarize
summarize(Path(os.environ['RESULT_DIR']))
PY
    then
      python3 code/summarize_herald_r1.py
      echo "COMPLETED $dataset $model $(date -u +%FT%TZ)"
    else
      echo "FAILED $dataset $model $(date -u +%FT%TZ)"
    fi
  done
done

HET_STAGE_ROOT=/dev/shm/rapid_het_holdout_d32_main_20261104 \
  HET_DATASETS=kaggle HET_MODELS=deepfm \
  HET_TAG=sgd_510_stable_bce HET_LR=0.01 \
  bash code/run_het_d32_holdout.sh
python3 code/summarize_het_stable_deepfm.py
python3 code/audit_r1_r5_final.py
