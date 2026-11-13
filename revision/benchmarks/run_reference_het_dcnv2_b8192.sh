#!/usr/bin/env bash
set -u

root=/home/longjiu.bty/Rapid/code
rate=${1:-1}
dim=${2:-64}
tag=store${rate//./}
if [[ ${RAPID_HET_GPU_INPUT:-0} == 1 ]]; then tag=${tag}_gpu; fi
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN

for dataset in ${RW_DATASETS:-terabyte kaggle}; do
  case "$dataset" in
    kaggle) stage=/dev/shm/rapid_shuffle2026/kaggle/het_v2 ;;
    terabyte) stage=/dev/shm/rapid_shuffle2026/terabyte/het_v3 ;;
  esac
  vocab=$(/tmp/rapid_torchrec11/bin/python - "$stage/partition_balanced.npz" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1]) as partition:
    print(len(partition['embed_partition']))
PY
)
  out=/tmp/rapid_shuffle2026_results/$dataset
  (
    cd "$root/upstream_fresh/HET-GMP" || exit 1
    RAPID_HET_DATA="$stage" RAPID_HET_VOCAB="$vocab" RAPID_HET_BIASES=1 \
      PYTHONPATH="$root/upstream_fresh/HET-GMP/python:/tmp/rapid_het_build/lib" \
      LD_LIBRARY_PATH="/tmp/rapid_het_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common" \
      OMP_NUM_THREADS=1 mpirun -np 8 /tmp/rapid_torchrec11/bin/python -u examples/hetuctr.py \
      --dataset criteo_rapid --model dcnv2 --batch_size 8192 --embed_dim "$dim" \
      --bound 0 --iter 510 --log_every 10 \
      --partition "$stage/partition_balanced.npz" --store_rate "$rate" --benchmark \
      --profile-io-prefix "$out/het_reference_verified_dcnv2_b8192_d${dim}_${tag}_510_io" \
      --output "$out/het_reference_verified_dcnv2_b8192_d${dim}_${tag}_510.metrics"
  ) > "$out/het_reference_verified_dcnv2_b8192_d${dim}_${tag}_510.log" 2>&1 || echo "FAILED HET-GMP $dataset rate=$rate dim=$dim" >&2
done
