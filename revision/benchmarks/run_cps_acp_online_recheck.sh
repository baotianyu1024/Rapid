#!/usr/bin/env bash
set -euo pipefail
root=/home/longjiu.bty/Rapid/code
out=/tmp/rapid_acp_recheck
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
export PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
export OMP_NUM_THREADS=8 HUGECTR_BUFFERED_IO=1 RAPID_IO_PROFILE=1
export RAPID_PLAN_SOURCE=/dev/shm/rapid_read_fair/kaggle/source.bin RAPID_PLAN_DIR=/tmp/rapid_fresh_stage/kaggle_final_plan RAPID_DENSE_COUNT=13 RAPID_MAPPING_THREADS=16
export RAPID_TABLE_LENGTHS=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1
for run in 1 2 3; do
 for mode in remote_acp forced_table20_reduction; do
  /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" --data "$RAPID_PLAN_SOURCE" --sizes-json /tmp/rapid_kaggle_plan_sizes.json --acp-plan "$out/$mode.json" --steps 110 --display 10 > "$out/online_${mode}_${run}.log" 2>&1
 done
done
