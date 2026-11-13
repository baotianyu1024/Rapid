#!/usr/bin/env bash
set -euo pipefail
root=/home/longjiu.bty/Rapid/code
stage=/dev/shm/rapid_read_fair/kaggle
out=/tmp/rapid_ram_fair/kaggle
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
export PYTHONPATH=/tmp/rapid_hugectr_build/lib
export LD_LIBRARY_PATH=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
export OMP_NUM_THREADS=8 HUGECTR_BUFFERED_IO=1 RAPID_CUDA_PROFILE_START=30 RAPID_CUDA_PROFILE_END=50
for mode in acp_only online; do
  if [[ $mode == acp_only ]]; then
    data=$stage/native.bin
    sizes=/tmp/rapid_fresh_stage/hugectr_controls/native_sizes.json
    plan=/tmp/rapid_kaggle_hash_acp.json
    unset RAPID_PLAN_SOURCE RAPID_PLAN_DIR RAPID_DENSE_COUNT RAPID_TABLE_LENGTHS RAPID_MAPPING_THREADS || true
  else
    data=$stage/source.bin
    sizes=/tmp/rapid_kaggle_plan_sizes.json
    plan=/tmp/rapid_kaggle_paper_acp.json
    export RAPID_PLAN_SOURCE=$data RAPID_PLAN_DIR=/tmp/rapid_fresh_stage/kaggle_final_plan RAPID_DENSE_COUNT=13
    export RAPID_TABLE_LENGTHS=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 RAPID_MAPPING_THREADS=16
  fi
  nsys profile --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx --sample=none -o "$out/${mode}_ram_profile" /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" --data "$data" --sizes-json "$sizes" --acp-plan "$plan" --steps 60 --display 10 > "$out/${mode}_ram_profile.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv "$out/${mode}_ram_profile.nsys-rep" > "$out/${mode}_ram_profile_stats.csv" 2> "$out/${mode}_ram_profile_stats.err"
done
