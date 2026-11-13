#!/usr/bin/env bash
set -euo pipefail
root=/home/longjiu.bty/Rapid/code
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
for dataset in kaggle terabyte taobao; do
  if [[ $dataset == taobao ]]; then
    script=torchrec_deepfm.py
    input=/dev/shm/rapid_read_fair/taobao/torch
  else
    script=torchrec_criteo.py
    input=/dev/shm/rapid_shuffle2026/$dataset/torch
  fi
  output=/tmp/rapid_shuffle2026_results/$dataset
  LD_LIBRARY_PATH="$common" OMP_NUM_THREADS=1 /tmp/rapid_torchrec11/bin/torchrun --standalone --nproc-per-node=8 "$root/$script" --data "$input" --input-mode prefetch --steps 1 --warmup 0 --out "$output/torch_plan_probe.json" --dump-plan "$output/torch_actual_plan.json" > "$output/torch_plan_probe.log" 2>&1
done
