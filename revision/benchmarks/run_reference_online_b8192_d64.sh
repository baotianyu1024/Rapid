#!/usr/bin/env bash
set -u

root=/home/longjiu.bty/Rapid/code
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
hlib=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
export RAPID_DENSE_COUNT=13 RAPID_TABLE_LENGTHS=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1
export RAPID_MAPPING_THREADS=16 HUGECTR_BUFFERED_IO=1 OMP_NUM_THREADS=8
export PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib"

for dataset in terabyte kaggle; do
  stage=/dev/shm/rapid_shuffle2026/$dataset
  out=/tmp/rapid_shuffle2026_results/$dataset
  case "$dataset" in
    kaggle) plan=/tmp/rapid_shuffle2026_kaggle_final_plan_fast ;;
    terabyte) plan=/tmp/rapid_shuffle2026_terabyte_final_plan ;;
  esac
  export RAPID_PLAN_SOURCE="$stage/torch/input.bin" RAPID_PLAN_DIR="$plan"
  /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" \
    --data "$RAPID_PLAN_SOURCE" --sizes-json "$out/plan_sizes.json" \
    --acp-plan "$plan/acp.json" --shape-profile reference --embed-dim 64 \
    --model dcnv2 --batch-size 65536 --steps 510 --display 100 \
    > "$out/rapid_online_reference_dcnv2_b8192_d64_510.log" 2>&1 || echo "FAILED online $dataset" >&2
done
