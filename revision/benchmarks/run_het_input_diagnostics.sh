#!/usr/bin/env bash
# Same balanced GMP batches: pageable CPU, pinned CPU, or GPU-resident sparse IDs.
set -euo pipefail
ds=${1:?dataset required}
case "$ds" in kaggle|terabyte|taobao) ;; *) exit 2;; esac
root=/home/longjiu.bty/Rapid/code
stage=/dev/shm/rapid_read_fair/$ds
out=/tmp/rapid_ram_fair/$ds
mkdir -p "$out"
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
if [[ $ds == kaggle ]]; then vocab=19279324; hetds=criteo_rapid; model=dcnv2; dim=32
elif [[ $ds == terabyte ]]; then vocab=21459313; hetds=criteo_rapid; model=dcnv2; dim=32
else vocab=964982; hetds=taobao; model=dfm; dim=128; fi
for mode in pageable pinned gpu; do
 for run in 1 2 3; do
  envargs=()
  [[ $mode == pinned ]] && envargs+=(RAPID_HET_PIN_INPUT=1)
  [[ $mode == gpu ]] && envargs+=(RAPID_HET_GPU_INPUT=1)
  if [[ $run == 1 ]]; then
   profile="$out/het_${mode}_profile"; mkdir -p "$profile"; envargs+=(RAPID_HET_INPUT_PROFILE_DIR="$profile")
  fi
  (cd "$root/upstream_fresh/HET-GMP"; env "${envargs[@]}" RAPID_HET_DATA="$stage/het" RAPID_HET_VOCAB="$vocab" RAPID_HET_BIASES=1 PYTHONPATH="$root/upstream_fresh/HET-GMP/python:/tmp/rapid_het_build/lib" LD_LIBRARY_PATH="/tmp/rapid_het_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common" OMP_NUM_THREADS=1 mpirun -np 8 /tmp/rapid_torchrec11/bin/python -u examples/hetuctr.py --dataset "$hetds" --model "$model" --batch_size 8192 --embed_dim "$dim" --bound 0 --iter 110 --log_every 10 --partition "$stage/het/partition_balanced.npz" --store_rate 1 --benchmark --profile-io-prefix "$out/het_${mode}_${run}_io" --output "$out/het_${mode}_${run}.metrics") > "$out/het_${mode}_${run}.log" 2>&1
 done
done
