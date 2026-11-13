#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
dataset="${RAPID_HERALD_DATASET:-taobao}"
model="${RAPID_HERALD_MODEL:-dlrm}"
steps="${RAPID_HERALD_STEPS:-10}"
extra_args=()
if [[ "${RAPID_HERALD_LOCAL_SHARED:-0}" == 1 ]]; then
  extra_args+=(--local-shared)
fi
run_root="${RAPID_HERALD_RUN_ROOT:-/tmp/rapid_herald_r1_20261105}"
out="$run_root/${dataset}_${model}_${steps}steps"
mkdir -p "$out"
export PYTHONPATH="$PWD/code/upstream_fresh/Herald/python:/tmp/rapid_herald_pydeps_pure:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/tmp/rapid_cuda12/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:/tmp/rapid_herald_deps_20261105/protobuf_install/lib64:/tmp/rapid_deps/usr/lib/x86_64-linux-gnu:/tmp/rapid_herald_cuda12_build_20261104/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export DMLC_INTERFACE=lo NCCL_SOCKET_IFNAME=lo NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export OMPI_MCA_btl_tcp_if_include=lo

python3 code/upstream_fresh/Herald/python/runner.py -w 8 -s 1 \
  python3 "$PWD/code/herald_r1.py" \
  --data "/dev/shm/rapid_rw_holdout_20261103/$dataset/native_train.bin" \
  --sizes-json "/tmp/rapid_rw_e2e_20261103/$dataset/native_sizes_verified.json" \
  --dataset "$dataset" --model "$model" --embed-dim "${RAPID_HERALD_DIM:-32}" \
  --steps "$steps" --out-dir "$out" \
  "${extra_args[@]}"
