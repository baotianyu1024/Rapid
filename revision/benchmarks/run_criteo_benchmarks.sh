#!/usr/bin/env bash
# Reproduce the Criteo engineering measurements in the existing /tmp environment.
set -euo pipefail
root=/home/longjiu.bty/Rapid/code
stage=/tmp/rapid_fresh_stage
py=/tmp/rapid_torchrec11/bin
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
for dataset in kaggle terabyte; do
  for run in 1 2 3; do
    (cd "$root"; LD_LIBRARY_PATH="$common" OMP_NUM_THREADS=1 "$py/torchrun" --standalone --nproc-per-node=8 torchrec_criteo.py --data "$stage/${dataset}_8/data" --steps 80 --warmup 10 --out "/tmp/rapid_torchrec_${dataset}_run${run}.json") > "/tmp/rapid_torchrec_${dataset}_run${run}.log" 2>&1
  done
done
for dataset in kaggle terabyte; do
  if [[ $dataset == kaggle ]]; then vocab=19279324; else vocab=21459313; fi
  for run in 1 2 3; do
    (cd "$root/upstream_fresh/HET-GMP"; RAPID_HET_DATA="$stage/het_${dataset}" RAPID_HET_VOCAB="$vocab" PYTHONPATH="$root/upstream_fresh/HET-GMP/python:/tmp/rapid_het_build/lib" LD_LIBRARY_PATH="/tmp/rapid_het_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common" OMP_NUM_THREADS=1 mpirun -np 8 "$py/python" -u examples/hetuctr.py --dataset criteo_rapid --model dcnv2 --batch_size 8192 --embed_dim 32 --bound 0 --iter 90 --log_every 10 --partition "$stage/het_${dataset}/partition.npz" --store_rate 1 --benchmark --output "/tmp/rapid_het_${dataset}_run${run}.metrics") > "/tmp/rapid_het_${dataset}_run${run}.log" 2>&1
  done
done
for dataset in kaggle terabyte; do
  for run in 1 2 3; do
    sourcefile="$stage/${dataset}_8_v2/data/input.bin"
    (cd "$root"; RAPID_PLAN_SOURCE="$sourcefile" RAPID_PLAN_DIR="$stage/${dataset}_final_plan" RAPID_DENSE_COUNT=13 RAPID_TABLE_LENGTHS=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common" OMP_NUM_THREADS=8 "$py/python" -u hugectr_criteo.py --data "$sourcefile" --plan "$stage/${dataset}_final_plan" --communication "$stage/${dataset}_8_v2/communication.json" --steps 90 --display 10) > "/tmp/rapid_hugectr_${dataset}_run${run}.log" 2>&1
  done
done
