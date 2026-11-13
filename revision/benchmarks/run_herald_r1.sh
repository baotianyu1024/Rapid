#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
for dataset in ${RAPID_HERALD_DATASETS:-taobao kaggle terabyte}; do
  for model in dlrm dcnv2 wide_deep deepfm; do
    out="/tmp/rapid_herald_r1_20261105/${dataset}_${model}_510steps"
    if [[ -s "$out/rank_0.json" ]] && RESULT_DIR="$out" python3 - <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0, 'code')
from summarize_herald_r1 import summarize
summarize(Path(os.environ['RESULT_DIR']))
PY
    then
      echo "SKIP completed $dataset $model"
      continue
    fi
    echo "START $dataset $model $(date -u +%FT%TZ)"
    RAPID_HERALD_DATASET="$dataset" RAPID_HERALD_MODEL="$model" \
      RAPID_HERALD_STEPS=510 RAPID_HERALD_LOCAL_SHARED=0 \
      timeout -k 15s 7200s bash code/probe_herald_r1.sh \
      > "/tmp/rapid_herald_r1_20261105/${dataset}_${model}_launcher.log" 2>&1
    RESULT_DIR="$out" python3 - <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0, 'code')
from summarize_herald_r1 import summarize
print(summarize(Path(os.environ['RESULT_DIR'])))
PY
    python3 code/summarize_herald_r1.py
    echo "END $dataset $model $(date -u +%FT%TZ)"
  done
done
