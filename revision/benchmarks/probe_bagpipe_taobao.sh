#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/upstream_fresh/BagPipe" && pwd)"
trainers="${RAPID_BAGPIPE_TRAINERS:-1}"
steps="${RAPID_BAGPIPE_STEPS:-2}"
dataset="${RAPID_BAGPIPE_DATASET:-taobao}"
model="${RAPID_BAGPIPE_MODEL:-dlrm}"
timeout_seconds="${RAPID_BAGPIPE_TIMEOUT:-3600}"
if [[ "$trainers" != 1 && "$trainers" != 8 ]]; then
  echo 'RAPID_BAGPIPE_TRAINERS must be 1 or 8' >&2
  exit 2
fi
case "$dataset" in
  taobao) dense_count=1 ;;
  kaggle|terabyte|t180|t200|t510) dense_count=13 ;;
  *) echo 'unsupported dataset' >&2; exit 2 ;;
esac
dim="${RAPID_BAGPIPE_DIM:-32}"
local_batch="${RAPID_BAGPIPE_LOCAL_BATCH:-8192}"
if [[ "$model" == wide_deep || "$model" == deepfm ]]; then
  emb_size="$((dim + 4))"
else
  emb_size="$dim"
fi
run_root="${RAPID_BAGPIPE_RUN_ROOT:-/tmp/rapid_bagpipe_20261105}"
work="$run_root/${dataset}_${model}_${trainers}gpu_${steps}steps"
if [[ "$trainers" == 8 ]]; then
  cache_size="${RAPID_BAGPIPE_CACHE_SIZE:-1000000}"
  lookahead="${RAPID_BAGPIPE_LOOKAHEAD:-8}"
else
  cache_size="${RAPID_BAGPIPE_CACHE_SIZE:-100000}"
  lookahead="${RAPID_BAGPIPE_LOOKAHEAD:-2}"
fi
cleanup_proportion="${RAPID_BAGPIPE_CLEANUP_PROPORTION:-0.25}"
available=$(df -B1 --output=avail /tmp | tail -1)
if (( available < 8589934592 )); then
  echo 'insufficient space for BagPipe output and reserve' >&2
  exit 1
fi
mkdir -p "$work"
python=/tmp/rapid_torchrec11/bin/python
"$python" - "$dataset" "$work/emb_info.txt" "${RAPID_TSERIES_DATA_ROOT:-/tmp/rapid_tseries_scaled_20261112}" <<'PY'
import json
import sys
from pathlib import Path
dataset = sys.argv[1]
path = (Path(sys.argv[3]) / dataset / 'sizes.json'
        if dataset.startswith('t') and dataset[1:].isdigit()
        else Path('/tmp/rapid_rw_e2e_20261103') / dataset / 'native_sizes_verified.json')
sizes = json.loads(path.read_text())
Path(sys.argv[2]).write_text(''.join(f'{size}\n' for size in sizes))
PY
input_dir="${RAPID_BAGPIPE_INPUT_DIR:-/dev/shm/rapid_bagpipe_input_20261105/$dataset}"
if [[ ! -s "$input_dir/part_0.txt" ]]; then
  echo "BagPipe input is missing: $input_dir/part_0.txt" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export BAGPIPE_LOG_DIR="$work"
if [[ "$trainers" == 8 ]]; then
  export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
  backend=nccl
  port=19185
else
  backend=gloo
  port=19175
fi
cd "$repo"

"$python" OracleCacher/oracle_cacher.py --worker-id 0 --world-size "$((trainers + 2))" \
  --world-size-trainers "$trainers" --master-ip 127.0.0.1 --master-port "$port" --mini-batch-size "$((local_batch * trainers))" \
  --stop-iter "$steps" --lookahead-value "$lookahead" --cache-size "$cache_size" --dataset-multi-num 1 \
  --cleanup-proportion "$cleanup_proportion" \
  --processed-csv "$input_dir" \
  --emb-info-file "$work/emb_info.txt" --ragged --repeat-cached-input \
  > "$work/oracle.log" 2>&1 &
oracle_pid=$!
"$python" EmbServer/embedding_server.py --worker-id 1 --world-size "$((trainers + 2))" \
  --master-ip 127.0.0.1 --master-port "$port" --emb-size "$emb_size" --emb-info-file "$work/emb_info.txt" \
  > "$work/embserver.log" 2>&1 &
server_pid=$!
trainer_pids=()
cleanup() {
  kill "$oracle_pid" "$server_pid" "${trainer_pids[@]}" 2>/dev/null || true
  wait "$oracle_pid" "$server_pid" "${trainer_pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT
for ((rank = 0; rank < trainers; rank++)); do
  CUDA_VISIBLE_DEVICES="$rank" timeout "$timeout_seconds" "$python" examples/trainer_main.py \
    --worker-id "$((rank + 2))" --world-size "$((trainers + 2))" \
    --dist-worker-id "$rank" --world-size-trainers "$trainers" \
    --master-ip 127.0.0.1 --dist-master-ip 127.0.0.1 --master-port "$port" --master-port-ccom "$((port + 1))" \
    --dist-backend "$backend" --device cuda:0 --emb-size "$emb_size" --cache-size "$cache_size" \
    --lookahead-value "$lookahead" --mini-batch-size "$local_batch" --stop-iter "$steps" --ragged \
    --cleanup-batch-proportion "$cleanup_proportion" \
    --model "$model" --dense-count "$dense_count" --optimizer adagrad --lr 0.01 \
    --emb-info-file "$work/emb_info.txt" --arch-mlp-bot 1-64-32 \
    --arch-mlp-top 64-32-1 > "$work/trainer_${rank}.log" 2>&1 &
  trainer_pids+=("$!")
done
result=0
for pid in "${trainer_pids[@]}"; do
  wait "$pid" || result=1
done
exit "$result"
