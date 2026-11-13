#!/usr/bin/env bash
# Diagnostic upper bound: after the first batch, reuse device input and skip reader/H2D.
set -euo pipefail
root=/home/longjiu.bty/Rapid/code
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
hlib=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
for dataset in kaggle terabyte; do
  stage=/dev/shm/rapid_shuffle2026/$dataset
  out=/tmp/rapid_shuffle2026_results/$dataset
  sizes=/tmp/rapid_shuffle2026_${dataset}_controls/native_sizes.json
  SKIP_H2D=1 HUGECTR_BUFFERED_IO=1 RAPID_IO_PROFILE=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" --data "$stage/native.bin" --sizes-json "$sizes" --acp-plan "$out/hash_acp.json" --steps 110 --display 10 > "$out/hugectr_acp_skip_h2d.log" 2>&1
done
