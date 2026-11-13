#!/usr/bin/env bash
# Serial, warm-cache engineering comparison; all paths are fresh downloads, not store.
set -euo pipefail
root=/home/longjiu.bty/Rapid/code
stage=/tmp/rapid_fresh_stage
out=/tmp/rapid_kaggle_controlled
mkdir -p "$out"
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
hlib=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
for file in "$stage/hugectr_controls/native_compact.bin" "$stage/hugectr_controls/rapid_materialized.bin" "$stage/kaggle_8_v2/data/input.bin" "$stage/kaggle_8/data/sparse.npy"; do
  dd if="$file" of=/dev/null bs=8M status=none
done
for variant in reduction unique acp_only materialized online; do
  for run in 1 2 3; do
    if [[ $variant == reduction || $variant == unique ]]; then
      data=$stage/hugectr_controls/native_compact.bin
      sizes=$stage/hugectr_controls/native_sizes.json
      strategy=(--strategy "$variant")
    elif [[ $variant == acp_only ]]; then
      data=$stage/hugectr_controls/native_compact.bin
      sizes=$stage/hugectr_controls/native_sizes.json
      strategy=(--acp-plan /tmp/rapid_kaggle_hash_acp.json)
    elif [[ $variant == materialized ]]; then
      data=$stage/hugectr_controls/rapid_materialized.bin
      sizes=/tmp/rapid_kaggle_plan_sizes.json
      strategy=(--acp-plan /tmp/rapid_kaggle_paper_acp.json)
    else
      data=$stage/kaggle_8_v2/data/input.bin
      sizes=/tmp/rapid_kaggle_plan_sizes.json
      strategy=(--acp-plan /tmp/rapid_kaggle_paper_acp.json)
    fi
    if [[ $variant == online ]]; then
      envargs=(RAPID_PLAN_SOURCE="$data" RAPID_PLAN_DIR="$stage/kaggle_final_plan" RAPID_DENSE_COUNT=13 RAPID_TABLE_LENGTHS=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 RAPID_MAPPING_THREADS=16)
    else
      envargs=()
    fi
    env "${envargs[@]}" HUGECTR_BUFFERED_IO=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" --data "$data" --sizes-json "$sizes" "${strategy[@]}" --steps 90 --display 10 > "$out/${variant}_${run}.log" 2>&1
  done
done
for run in 1 2 3; do
  LD_LIBRARY_PATH="$common" OMP_NUM_THREADS=1 /tmp/rapid_torchrec11/bin/torchrun --standalone --nproc-per-node=8 "$root/torchrec_criteo.py" --data "$stage/kaggle_8/data" --steps 80 --warmup 10 --out "$out/torchrec_${run}.json" > "$out/torchrec_${run}.log" 2>&1
done
for run in 1 2 3; do
  (cd "$root/upstream_fresh/HET-GMP"; RAPID_HET_DATA="$stage/het_kaggle" RAPID_HET_VOCAB=19279324 PYTHONPATH="$root/upstream_fresh/HET-GMP/python:/tmp/rapid_het_build/lib" LD_LIBRARY_PATH="/tmp/rapid_het_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common" OMP_NUM_THREADS=1 mpirun -np 8 /tmp/rapid_torchrec11/bin/python -u examples/hetuctr.py --dataset criteo_rapid --model dcnv2 --batch_size 8192 --embed_dim 32 --bound 0 --iter 90 --log_every 10 --partition "$stage/het_kaggle/partition.npz" --store_rate 1 --benchmark --output "$out/het_${run}.metrics") > "$out/het_${run}.log" 2>&1
done
