#!/usr/bin/env bash
set -euo pipefail
dataset=${1:?kaggle or terabyte}
case "$dataset" in kaggle|terabyte) ;; *) exit 2 ;; esac
root=/home/longjiu.bty/Rapid/code
if [[ $dataset == kaggle ]]; then version=het_v2; else version=het_v3; fi
stage=/dev/shm/rapid_shuffle2026/$dataset/$version
out=/tmp/rapid_shuffle2026_results/$dataset
mkdir -p "$out"
vocab=$(python - "$stage/partition_balanced.npz" <<'PY'
import sys,numpy as np
with np.load(sys.argv[1]) as data:
    print(len(data['embed_partition']))
PY
)
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
for mode in pageable gpu; do
  for run in 1 2 3; do
    envargs=()
    [[ $mode == gpu ]] && envargs+=(RAPID_HET_GPU_INPUT=1)
    (cd "$root/upstream_fresh/HET-GMP"; env "${envargs[@]}" RAPID_HET_DATA="$stage" RAPID_HET_VOCAB="$vocab" RAPID_HET_BIASES=1 PYTHONPATH="$root/upstream_fresh/HET-GMP/python:/tmp/rapid_het_build/lib" LD_LIBRARY_PATH="/tmp/rapid_het_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common" OMP_NUM_THREADS=1 mpirun -np 8 /tmp/rapid_torchrec11/bin/python -u examples/hetuctr.py --dataset criteo_rapid --model dcnv2 --batch_size 8192 --embed_dim 32 --bound 0 --iter 110 --log_every 10 --partition "$stage/partition_balanced.npz" --store_rate 1 --benchmark --profile-io-prefix "$out/het_${mode}_${run}_io" --output "$out/het_${mode}_${run}.metrics") > "$out/het_${mode}_${run}.log" 2>&1
  done
done
