#!/usr/bin/env bash
set -euo pipefail

seed=${1:-41}
steps=${2:-730}
root=/home/longjiu.bty/Rapid
plan=/dev/shm/rapid_taobao_full_auc_20261109/rapid
out=/tmp/rapid_taobao_full_auc_20261109
test -s "$plan/train.bin"
test -s "$plan/eval.bin"
test -s "$plan/embedding_mapping.tsv"
mkdir -p "$out"

env LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 \
  PYTHONPATH=/tmp/rapid_hugectr_build/lib \
  LD_LIBRARY_PATH=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64 \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN \
  HUGECTR_BUFFERED_IO=1 OMP_NUM_THREADS=8 \
  /tmp/rapid_torchrec11/bin/python -u "$root/code/hugectr_criteo.py" \
  --data "$plan/train.bin" --eval-data "$plan/eval.bin" \
  --train-samples 23920640 --eval-samples 2621440 --max-eval-batches 40 \
  --plan "$plan" --acp-plan "$plan/acp.json" \
  --dense-count 1 --hotness 1,1,1,1 --shape-profile reference \
  --embed-dim 32 --model dlrm --batch-size 65536 \
  --steps "$steps" --display 365 --eval-interval 365 \
  --eval-after-fit --zero-embedding-init \
  --optimizer adagrad --lr 0.01 --seed "$seed" \
  > "$out/rapid_seed${seed}_${steps}.log" 2>&1
