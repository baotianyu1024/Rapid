#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
mkdir -p /tmp/rapid_bagpipe_20261105
for dataset in kaggle terabyte; do
  for model in dlrm dcnv2 wide_deep deepfm; do
    if [[ "$dataset" == kaggle && "$model" == dlrm ]]; then
      continue
    fi
    result_dir="/tmp/rapid_bagpipe_20261105/${dataset}_${model}_8gpu_510steps"
    if [[ -d "$result_dir" ]] && RESULT_DIR="$result_dir" python3 - <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0, 'code')
from summarize_bagpipe_r1 import summarize
summarize(Path(os.environ['RESULT_DIR']))
PY
    then
      echo "SKIP completed $dataset $model"
      continue
    fi
    echo "START $dataset $model $(date -u +%FT%TZ)"
    RAPID_BAGPIPE_DATASET="$dataset" RAPID_BAGPIPE_MODEL="$model" \
      RAPID_BAGPIPE_TRAINERS=8 RAPID_BAGPIPE_STEPS=510 \
      RAPID_BAGPIPE_CACHE_SIZE=16000000 \
      bash code/probe_bagpipe_taobao.sh
    result=$?
    echo "END $dataset $model status=$result $(date -u +%FT%TZ)"
    if ! RESULT_DIR="$result_dir" python3 - <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0, 'code')
from summarize_bagpipe_r1 import summarize
summarize(Path(os.environ['RESULT_DIR']))
PY
    then
      exit "${result:-1}"
    fi
    python3 code/summarize_bagpipe_r1.py || exit $?
  done
done
