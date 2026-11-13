#!/usr/bin/env bash
set -euo pipefail

root=/home/longjiu.bty/Rapid/code
out=/tmp/rapid_rw_d32_input_probe_20261104
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export LD_LIBRARY_PATH=$common
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
mkdir -p "$out"

for dataset in terabyte kaggle taobao movielens25m_user_temporal; do
  case "$dataset" in
    terabyte|kaggle)
      data=/dev/shm/rapid_shuffle2026/$dataset/torch; model=dcnv2 ;;
    taobao)
      data=/dev/shm/rapid_read_fair/taobao/torch; model=dcnv2 ;;
    movielens25m_user_temporal)
      data=/dev/shm/rapid_movielens25m_user_temporal_20261103/train; model=dlrm ;;
  esac
  for mode in prefetch gpu; do
    output=$out/${dataset}_${model}_${mode}.json
    if [[ -s $output ]]; then continue; fi
    OMP_NUM_THREADS=1 /tmp/rapid_torchrec11/bin/torchrun --standalone --nproc-per-node=8 \
      "$root/torchrec_criteo.py" --data "$data" --input-mode "$mode" \
      --row-wise-only --shape-profile reference --embed-dim 32 --local-batch 8192 \
      --model "$model" --steps 100 --warmup 10 --optimizer adagrad --lr 0.01 \
      --dump-plan "$out/${dataset}_${model}_${mode}_plan.json" --out "$output" \
      > "$out/${dataset}_${model}_${mode}.log" 2>&1
    echo "completed input probe $dataset $mode" >&2
  done
done
