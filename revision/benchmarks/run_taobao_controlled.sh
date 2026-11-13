#!/usr/bin/env bash
set -euo pipefail
root=/home/longjiu.bty/Rapid/code
stage=/tmp/rapid_fresh_stage
out=/tmp/rapid_taobao_controlled
mkdir -p "$out"
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
hlib=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
for file in "$stage/taobao_controls/native_compact.bin" "$stage/taobao_controls/materialized.bin" "$stage/taobao_8_v2/data/input.bin" "$stage/taobao_8_v2/data/sparse.npy"; do
  dd if="$file" of=/dev/null bs=8M status=none
done
for variant in native materialized online; do
 for run in 1 2 3; do
  if [[ $variant == native ]]; then
   data=$stage/taobao_controls/native_compact.bin
   sizes=(413951 249351 179015 122665)
   strategy=(--unique)
  else
   sizes=(419777 253686 180998 125263)
   strategy=(--acp-plan /tmp/rapid_taobao_paper_acp.json)
   if [[ $variant == materialized ]]; then data=$stage/taobao_controls/materialized.bin; else data=$stage/taobao_8_v2/data/input.bin; fi
  fi
  if [[ $variant == online ]]; then
   envargs=(RAPID_PLAN_SOURCE="$data" RAPID_PLAN_DIR="$stage/taobao_final_plan" RAPID_DENSE_COUNT=1 RAPID_TABLE_LENGTHS=1,1,1,1 RAPID_MAPPING_THREADS=16)
  else envargs=(); fi
  env "${envargs[@]}" HUGECTR_BUFFERED_IO=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_smoke.py" --data "$data" --batch-size 65536 --num-samples 1048576 --sizes "${sizes[@]}" "${strategy[@]}" --deepfm --steps 90 --display 10 > "$out/${variant}_${run}.log" 2>&1
done
done
for run in 1 2 3; do
 LD_LIBRARY_PATH="$common" OMP_NUM_THREADS=1 /tmp/rapid_torchrec11/bin/torchrun --standalone --nproc-per-node=8 "$root/torchrec_deepfm.py" --data "$stage/taobao_8_v2/data" --steps 80 --warmup 10 --out "$out/torchrec_${run}.json" > "$out/torchrec_${run}.log" 2>&1
done
for run in 1 2 3; do
 (cd "$root/upstream_fresh/HET-GMP"; RAPID_HET_DATA="$stage/het_taobao" RAPID_HET_VOCAB=964982 RAPID_HET_BIASES=1 PYTHONPATH="$root/upstream_fresh/HET-GMP/python:/tmp/rapid_het_build/lib" LD_LIBRARY_PATH="/tmp/rapid_het_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common" OMP_NUM_THREADS=1 mpirun -np 8 /tmp/rapid_torchrec11/bin/python -u examples/hetuctr.py --dataset taobao --model dfm --batch_size 8192 --embed_dim 128 --bound 0 --iter 90 --log_every 10 --partition "$stage/het_taobao/partition.npz" --store_rate 1 --benchmark --output "$out/het_${run}.metrics") > "$out/het_${run}.log" 2>&1
done
