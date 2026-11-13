#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
dataset="${1:?dataset t180, t200, or t510}"
model="${2:?model dlrm, dcnv2, wide_deep, or deepfm}"
steps="${3:-10}"
case "$dataset" in
  t180) batch=128; data=input_small.bin ;;
  t200|t510) batch=512; data=input.bin ;;
  *) echo "unsupported dataset: $dataset" >&2; exit 2 ;;
esac
source_dir="${RAPID_TSERIES_DATA_ROOT:-/tmp/rapid_tseries_scaled_20261112}/$dataset"
batch=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["local_batch"])' "$source_dir/manifest.json")
if [[ "$dataset" == t180 && -s "$source_dir/input.bin" && ! -e "$source_dir/input_small.bin" ]]; then
  data=input.bin
fi
out="${RAPID_HERALD_T_RUN_ROOT:-/tmp/rapid_herald_tseries_20261113}/${dataset}_${model}_${steps}steps"
available=$(df -B1 --output=avail /tmp | tail -1)
if (( available < 8589934592 )); then
  echo 'insufficient space for Herald output and reserve' >&2
  exit 1
fi
mkdir -p "$out"
export PYTHONPATH="$PWD/code/upstream_fresh/Herald/python:/tmp/rapid_herald_pydeps_pure:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/tmp/rapid_cuda12/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:/tmp/rapid_herald_deps_20261105/protobuf_install/lib64:/tmp/rapid_deps/usr/lib/x86_64-linux-gnu:/tmp/rapid_herald_cuda12_build_20261104/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export DMLC_INTERFACE=lo NCCL_SOCKET_IFNAME=lo NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export OMPI_MCA_btl_tcp_if_include=lo

python3 code/upstream_fresh/Herald/python/runner.py -w 8 -s 1 \
  python3 "$PWD/code/herald_r1.py" \
  --data "$source_dir/$data" --sizes-json "$source_dir/sizes.json" \
  --hotness-json "$source_dir/manifest.json" --local-batch "$batch" \
  --dataset "$dataset" --model "$model" --embed-dim 32 \
  --steps "$steps" --out-dir "$out"
