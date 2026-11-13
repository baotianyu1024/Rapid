#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

steps=${1:?number of training steps}
root=/home/longjiu.bty/Rapid/code
data=/dev/shm/rapid_movielens25m_user_temporal_20261103
out=/tmp/rapid_rw_e2e_20261103/movielens25m_user_temporal
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
hlib=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN

for model in dlrm neumf; do
  native_log=$out/hugectr_unique_${model}_b8192_d64_${steps}_adagrad.log
  torch_json=$out/torchrec_rowwise_${model}_b8192_d64_${steps}_adagrad.json
  if ! rg -q '^EVAL_METRICS=' "$native_log" 2>/dev/null; then
    HUGECTR_BUFFERED_IO=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 \
      /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" \
      --data "$data/train/input.bin" --eval-data "$data/eval/input.bin" \
      --train-samples 19922944 --eval-samples 5046272 --max-eval-batches 77 --eval-after-fit \
      --dense-count 1 --hotness 1,1 --sizes-json "$data/table_sizes.json" --strategy unique \
      --shape-profile reference --embed-dim 64 --model "$model" \
      --batch-size 65536 --steps "$steps" --display 100 --lr 0.01 --optimizer adagrad \
      > "$native_log" 2>&1
  fi
  if ! rg -q '"eval_history"' "$torch_json" 2>/dev/null; then
    LD_LIBRARY_PATH="$common" OMP_NUM_THREADS=1 \
      /tmp/rapid_torchrec11/bin/torchrun --standalone --nproc-per-node=8 \
      "$root/torchrec_criteo.py" --data "$data/train" --eval-data "$data/eval" \
      --eval-interval 1000000 --input-mode prefetch --row-wise-only \
      --shape-profile reference --embed-dim 64 --local-batch 8192 \
      --model "$model" --steps "$steps" --warmup 0 --lr 0.01 --optimizer adagrad \
      --dump-plan "$out/torchrec_rowwise_${model}_${steps}_adagrad_plan.json" \
      --out "$torch_json" > "${torch_json%.json}.log" 2>&1
  fi
done
