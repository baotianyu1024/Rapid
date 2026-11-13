#!/usr/bin/env bash
set -euo pipefail

root=/home/longjiu.bty/Rapid/code
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$common"
TIMEFORMAT='OUTER_PROCESS_SECONDS=%R'
{ time /tmp/rapid_torchrec11/bin/torchrun --standalone --nproc-per-node=8 \
    "$root/torchrec_deepfm.py" --data /tmp/rapid_fresh_stage/taobao_8_v2/data \
    --shape-profile reference --embed-dim 64 --local-batch 8192 \
    --input-mode prefetch --steps 510 --warmup 0 \
    --out /tmp/rapid_taobao_reference_deepfm_b8192_d64_torchrec_fixed_510.json; } \
    > /tmp/rapid_taobao_reference_deepfm_b8192_d64_torchrec_fixed_510.log 2>&1
