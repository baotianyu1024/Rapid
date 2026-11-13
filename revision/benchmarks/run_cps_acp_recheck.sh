#!/usr/bin/env bash
set -euo pipefail
root=/home/longjiu.bty/Rapid/code
out=/tmp/rapid_acp_recheck
mkdir -p "$out"
python "$root/Rapid/experiments/acp/export_hugectr_acp.py" --profile /tmp/rapid_fresh_stage/kaggle_8_v2/optimized/peak_samples/cps_metrics.json --output "$out/remote_acp.json" --cost-model remote
python - "$out" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
plan = json.loads((out / 'remote_acp.json').read_text())
assert all(t['strategy'] == 'Unique' for t in plan['tables'])
plan['tables'][20]['strategy'] = 'Reduction'
plan['selection'] = 'controlled experiment: same CPS data and ACP, table 20 forced Reduction'
(out / 'forced_table20_reduction.json').write_text(json.dumps(plan, indent=2) + '\n')
PY
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
export PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
export OMP_NUM_THREADS=8 HUGECTR_BUFFERED_IO=1 RAPID_IO_PROFILE=1
for run in 1 2 3; do
 for mode in remote_acp forced_table20_reduction; do
  /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" --data /dev/shm/rapid_read_fair/kaggle/materialized.bin --sizes-json /tmp/rapid_kaggle_plan_sizes.json --acp-plan "$out/$mode.json" --steps 110 --display 10 > "$out/${mode}_${run}.log" 2>&1
 done
done
