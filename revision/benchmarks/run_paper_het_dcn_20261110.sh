#!/usr/bin/env bash
set -euo pipefail

root=/home/longjiu.bty/Rapid
stage=/dev/shm/rapid_het_holdout_d32_20261104/kaggle
out=/tmp/rapid_paper_table1_20261110
vocab=$(/tmp/rapid_torchrec11/bin/python - "$stage/partition_train.npz" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1]) as partition:
    print(len(partition['embed_partition']))
PY
)
steps=${RUN_STEPS:-510}
lr=${RAPID_PAPER_HET_LR:-0.01}
tag=${RAPID_PAPER_HET_TAG:-init001}
log="$out/kaggle_dcn_gmp_${steps}_${tag}.log"
metrics="$out/kaggle_dcn_gmp_${steps}_${tag}.metrics"
if [[ -e "$log" ]]; then
  echo "Refusing to overwrite $log" >&2
  exit 1
fi
cd "$root/code/upstream_fresh/HET-GMP"
RAPID_HET_DATA="$stage/train" RAPID_HET_VOCAB="$vocab" \
  RAPID_HET_HOTNESS=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
  RAPID_HET_DENSE_DIM=13 RAPID_HET_LR="$lr" RAPID_HET_BIASES=1 \
  PYTHONPATH="$PWD/python:/tmp/rapid_het_build/lib" \
  LD_LIBRARY_PATH=/tmp/rapid_het_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64 \
  LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 \
  timeout 1800 mpirun -np 8 /tmp/rapid_torchrec11/bin/python -u examples/hetuctr.py \
    --dataset criteo_rapid --model dcn_multi --batch_size 8192 --embed_dim 32 \
    --bound 0 --iter "$steps" --log_every 10 --partition "$stage/partition_train.npz" \
    --store_rate 0.25 --benchmark --output "$metrics" > "$log" 2>&1
rg '^TRAIN' "$log" | tail -3
