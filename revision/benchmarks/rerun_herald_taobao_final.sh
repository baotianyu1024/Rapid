#!/usr/bin/env bash
set -euo pipefail

queue_pid="${RAPID_FINAL_QUEUE_PID:?set RAPID_FINAL_QUEUE_PID}"
while kill -0 "$queue_pid" 2>/dev/null; do
  sleep 30
done

cd "$(dirname "$0")/.."
for model in dlrm dcnv2 wide_deep deepfm; do
  out="/tmp/rapid_herald_r1_20261105/taobao_${model}_510steps"
  if [[ -d "$out" ]]; then
    mv "$out" "${out}_preopt_$(date -u +%Y%m%dT%H%M%S)"
  fi
  RAPID_HERALD_DATASET=taobao RAPID_HERALD_MODEL="$model" \
    RAPID_HERALD_STEPS=510 RAPID_HERALD_LOCAL_SHARED=0 \
    timeout -k 15s 1800s bash code/probe_herald_r1.sh \
    > "/tmp/rapid_herald_r1_20261105/taobao_${model}_final_launcher.log" 2>&1
  RESULT_DIR="$out" python3 - <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0, 'code')
from summarize_herald_r1 import summarize
print(summarize(Path(os.environ['RESULT_DIR'])))
PY
  python3 code/summarize_herald_r1.py
done
python3 code/audit_r1_r5_final.py
