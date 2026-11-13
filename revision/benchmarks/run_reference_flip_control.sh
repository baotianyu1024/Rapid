#!/usr/bin/env bash
set -u

root=/home/longjiu.bty/Rapid/code
dataset=${1:-terabyte}
out=/tmp/rapid_shuffle2026_results/$dataset
stage=/dev/shm/rapid_shuffle2026/$dataset
case "$dataset" in
  kaggle) plan=/tmp/rapid_shuffle2026_kaggle_final_plan_fast ;;
  terabyte) plan=/tmp/rapid_shuffle2026_terabyte_final_plan ;;
  *) echo "dataset must be kaggle or terabyte" >&2; exit 2 ;;
esac
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
hlib=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN

for dim_run in 32:1 64:2 64:3; do
  dim=${dim_run%:*}
  run=${dim_run#*:}
  rapid_log=$out/rapid_reference_dcnv2_b8192_d${dim}_repeat${run}_510.log
  torch_log=$out/torchrec_reference_dcnv2_b8192_d${dim}_repeat${run}_510.log
  torch_json=$out/torchrec_reference_dcnv2_b8192_d${dim}_repeat${run}_510.json
  if ! rg -q '^FIT_SECONDS=' "$rapid_log" 2>/dev/null; then
    HUGECTR_BUFFERED_IO=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 \
      /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" \
      --data "$stage/materialized.bin" --sizes-json "$out/plan_sizes.json" \
      --acp-plan "$plan/acp.json" --shape-profile reference --embed-dim "$dim" \
      --model dcnv2 --batch-size 65536 --steps 510 --display 100 \
      > "$rapid_log" 2>&1 || echo "FAILED rapid dim=$dim run=$run" >&2
  fi
  if rg -q '"loop_wall_seconds"' "$torch_json" 2>/dev/null; then
    continue
  fi
  TIMEFORMAT='OUTER_PROCESS_SECONDS=%R'
  { time LD_LIBRARY_PATH="$common" OMP_NUM_THREADS=1 \
    /tmp/rapid_torchrec11/bin/torchrun --standalone --nproc-per-node=8 \
    "$root/torchrec_criteo.py" --data "$stage/torch" --input-mode prefetch \
    --shape-profile reference --embed-dim "$dim" --local-batch 8192 \
    --model dcnv2 --steps 510 --warmup 0 --out "$torch_json"; } \
    > "$torch_log" 2>&1 || echo "FAILED torchrec dim=$dim run=$run" >&2
done
