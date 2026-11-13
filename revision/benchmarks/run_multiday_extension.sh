#!/usr/bin/env bash
set -euo pipefail

method=${1:?usage: run_multiday_extension.sh torchrec|rapid [steps] [eval|noeval]}
steps=${2:-6500}
eval_mode=${3:-eval}
root=/home/longjiu.bty/Rapid
case_dir=/tmp/rapid_criteo_multiday_20261103
ram_dir=/dev/shm/rapid_criteo_multiday_20261103
test -s "$ram_dir/train.bin"
test -s "$ram_dir/native_eval.bin"
test -s "$ram_dir/rapid_eval.bin"
df -h /tmp /dev/shm
cd "$root"
case "$eval_mode" in
  eval) torch_eval=(--eval-data "$ram_dir/eval_torchrec" --eval-interval 100)
        huge_eval=(--eval-data "$ram_dir/rapid_eval.bin" --eval-samples 262144 --max-eval-batches 16 --eval-interval 100) ;;
  noeval) torch_eval=() huge_eval=(--eval-interval 1000000) ;;
  *) echo "unknown evaluation mode: $eval_mode" >&2; exit 2 ;;
esac
suffix=$steps
if [[ $eval_mode == noeval ]]; then suffix="${steps}_noeval"; fi

case "$method" in
  torchrec)
    env LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 \
      LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      /tmp/rapid_v100_venv/bin/torchrun --standalone --nproc_per_node=8 \
      code/torchrec_criteo.py --data "$ram_dir/train_torchrec" \
      "${torch_eval[@]}" \
      --steps "$steps" --warmup 0 --local-batch 2048 --embed-dim 128 \
      --lr 0.01 --input-mode prefetch --zero-embedding-init \
      --out "$case_dir/torchrec_ram_${suffix}.json" \
      > "$case_dir/torchrec_ram_${suffix}.log" 2>&1
    ;;
  rapid)
    env LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 \
      NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
      RAPID_PLAN_SOURCE="$ram_dir/train.bin" \
      RAPID_PLAN_DIR="$ram_dir/runtime_plan" RAPID_DENSE_COUNT=13 \
      RAPID_TABLE_LENGTHS=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
      RAPID_MAPPING_THREADS=16 HUGECTR_BUFFERED_IO=1 RAPID_IO_PROFILE=1 \
      PYTHONPATH=/tmp/rapid_hugectr_build/lib \
      LD_LIBRARY_PATH=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:/usr/local/nvidia/lib64 \
      OMP_NUM_THREADS=8 /tmp/rapid_v100_venv/bin/python -u \
      code/hugectr_criteo.py --data "$ram_dir/train.bin" \
      --plan "$ram_dir/runtime_plan" --acp-plan "$ram_dir/runtime_plan/acp.json" \
      --train-samples 1048576 "${huge_eval[@]}" --batch-size 16384 \
      --embed-dim 128 --lr 0.01 --steps "$steps" \
      --display 100 --zero-embedding-init \
      > "$case_dir/hugectr_rapid_ram_${suffix}.log" 2>&1
    ;;
  *) echo "unknown method: $method" >&2; exit 2 ;;
esac
