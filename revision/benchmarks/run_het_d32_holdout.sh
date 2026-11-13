#!/usr/bin/env bash
set -euo pipefail

base=/home/longjiu.bty/Rapid/code/upstream_fresh/HET-GMP
stage=${HET_STAGE_ROOT:-/dev/shm/rapid_het_holdout_d32_20261104}
out=/tmp/rapid_het_d32_e2e_20261104
tag=${HET_TAG:-sgd_510}
steps=${HET_STEPS:-510}
eval_every=${HET_EVAL_EVERY:-500}
learning_rate=${HET_LR:-0.01}
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
mkdir -p "$out"
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN

for dataset in ${HET_DATASETS:-terabyte kaggle taobao}; do
  case "$dataset" in
    taobao) name=taobao; dense=1; hot=1,1,1,1 ;;
    *) name=criteo_rapid; dense=13; hot=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 ;;
  esac
  part=$stage/$dataset/partition_train.npz
  vocab=$(/tmp/rapid_torchrec11/bin/python - "$part" <<'PY'
import numpy as np
import sys
with np.load(sys.argv[1]) as data:
    print(len(data['embed_partition']))
PY
)
  for model in ${HET_MODELS:-dcnv2}; do
    case "$model" in
      dcnv2) het_model=dcnv2 ;;
      dlrm) het_model=dlrm_multi ;;
      wide_deep) het_model=wdl_multi ;;
      deepfm) het_model=dfm_multi ;;
      *) echo "unsupported HET model: $model" >&2; exit 2 ;;
    esac
    log=$out/${dataset}_${model}_b8192_d32_${tag}.log
    metrics=$out/${dataset}_${model}_b8192_d32_${tag}.metrics
    if [[ -f $metrics ]] && rg -q '^EVAL_METRICS=' "$log"; then continue; fi
    (
    cd "$base"
    env RAPID_HET_DATA="$stage/$dataset/train" RAPID_HET_VAL_DATA="$stage/$dataset/valid" \
      RAPID_HET_VOCAB="$vocab" RAPID_HET_HOTNESS="$hot" RAPID_HET_DENSE_DIM="$dense" \
      RAPID_HET_LR="$learning_rate" RAPID_HET_BIASES=1 \
      PYTHONPATH="$base/python:/tmp/rapid_het_build/lib" \
      LD_LIBRARY_PATH="/tmp/rapid_het_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common" \
      OMP_NUM_THREADS=1 mpirun -np 8 /tmp/rapid_torchrec11/bin/python -u examples/hetuctr.py \
      --dataset "$name" --model "$het_model" --batch_size 8192 --embed_dim 32 \
      --bound 0 --iter "$steps" --log_every 10 --eval_every "$eval_every" \
      --partition "$part" --store_rate 0.25 --benchmark --val \
      --output "$metrics"
    ) > "$log" 2>&1
    echo "completed HET-GMP $dataset $model" >&2
  done
done
