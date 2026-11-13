#!/usr/bin/env bash
set -euo pipefail
dataset=${1:?kaggle or terabyte}
case "$dataset" in
  kaggle) plan=/tmp/rapid_shuffle2026_kaggle_final_plan_fast ;;
  terabyte) plan=/tmp/rapid_shuffle2026_terabyte_final_plan ;;
  *) exit 2 ;;
esac
root=/home/longjiu.bty/Rapid/code
stage=/dev/shm/rapid_shuffle2026/$dataset
out=/tmp/rapid_shuffle2026_results/$dataset
sizes=$out/plan_sizes.json
[[ -f $plan/manifest.json && -f $sizes && -f $stage/materialized.bin ]]
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
hlib=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
for variant in materialized online; do
  for run in 1 2 3; do
    if [[ $variant == online ]]; then
      data=$stage/torch/input.bin
      envargs=(RAPID_PLAN_SOURCE="$data" RAPID_PLAN_DIR="$plan" RAPID_DENSE_COUNT=13 RAPID_TABLE_LENGTHS=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 RAPID_MAPPING_THREADS=16)
    else
      data=$stage/materialized.bin
      envargs=()
    fi
    env "${envargs[@]}" HUGECTR_BUFFERED_IO=1 RAPID_IO_PROFILE=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" --data "$data" --sizes-json "$sizes" --acp-plan "$plan/acp.json" --steps 110 --display 10 > "$out/rapid_${variant}_${run}.log" 2>&1
  done
done
