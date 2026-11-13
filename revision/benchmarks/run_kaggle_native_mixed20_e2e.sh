#!/usr/bin/env bash
set -u
ulimit -c 0

root=/home/longjiu.bty/Rapid/code
split=/dev/shm/rapid_rw_holdout_20261103/kaggle
out=/tmp/rapid_rw_e2e_20261103/kaggle
sizes=$out/native_sizes_verified.json
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
hlib=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN

for model in dcnv2 dlrm wide_deep deepfm; do
  log=$out/hugectr_mixed20_verified_${model}_b8192_d64_510.log
  if rg -q '^EVAL_METRICS=' "$log" 2>/dev/null; then
    continue
  fi
  HUGECTR_BUFFERED_IO=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 \
    /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" \
    --data "$split/native_train.bin" --eval-data "$split/native_eval.bin" \
    --train-samples 983040 --eval-samples 65536 --eval-after-fit \
    --sizes-json "$sizes" --strategy mixed --reduction-tables 20 \
    --shape-profile reference --embed-dim 64 --model "$model" \
    --batch-size 65536 --steps 510 --display 100 \
    > "$log" 2>&1 || echo "FAILED Kaggle native Mixed20 $model" >&2
done
