#!/usr/bin/env bash
# Serial measurements; source files are checksum-verified tmpfs copies.
set -euo pipefail
ds=${1:?dataset required: kaggle, terabyte or taobao}
case "$ds" in kaggle|terabyte|taobao) ;; *) exit 2;; esac
root=/home/longjiu.bty/Rapid/code
stage=/dev/shm/rapid_read_fair/$ds
fresh=/tmp/rapid_fresh_stage
out=/tmp/rapid_ram_fair/$ds
mkdir -p "$out"
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
hlib=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
if [[ $ds == taobao ]]; then variants=(native materialized online); else variants=(reduction unique acp_only materialized online); fi
for variant in "${variants[@]}"; do
 for run in 1 2 3; do
  data=$stage/native.bin
  envargs=()
  if [[ $ds == taobao ]]; then
   if [[ $variant == native ]]; then sizes=(413951 249351 179015 122665); strategy=(--unique)
   else sizes=(419777 253686 180998 125263); strategy=(--acp-plan /tmp/rapid_taobao_paper_acp.json)
     if [[ $variant == materialized ]]; then data=$stage/materialized.bin; else data=$stage/source.bin; fi
   fi
   if [[ $variant == online ]]; then envargs=(RAPID_PLAN_SOURCE="$data" RAPID_PLAN_DIR="$fresh/taobao_final_plan" RAPID_DENSE_COUNT=1 RAPID_TABLE_LENGTHS=1,1,1,1 RAPID_MAPPING_THREADS=16); fi
   env "${envargs[@]}" HUGECTR_BUFFERED_IO=1 RAPID_IO_PROFILE=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_smoke.py" --data "$data" --batch-size 65536 --num-samples 1048576 --sizes "${sizes[@]}" "${strategy[@]}" --deepfm --steps 110 --display 10 > "$out/${variant}_${run}.log" 2>&1
  else
   sizes="$fresh/hugectr_controls/native_sizes.json"
   [[ $ds == terabyte ]] && sizes="$fresh/hugectr_terabyte_controls/native_sizes.json"
   if [[ $variant == reduction || $variant == unique ]]; then strategy=(--strategy "$variant")
   elif [[ $variant == acp_only ]]; then strategy=(--acp-plan "/tmp/rapid_${ds}_hash_acp.json")
   else sizes="/tmp/rapid_${ds}_plan_sizes.json"; strategy=(--acp-plan "/tmp/rapid_${ds}_paper_acp.json")
    if [[ $variant == materialized ]]; then data=$stage/materialized.bin; else data=$stage/source.bin; fi
   fi
   if [[ $variant == online ]]; then envargs=(RAPID_PLAN_SOURCE="$data" RAPID_PLAN_DIR="$fresh/${ds}_final_plan" RAPID_DENSE_COUNT=13 RAPID_TABLE_LENGTHS=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 RAPID_MAPPING_THREADS=16); fi
   env "${envargs[@]}" HUGECTR_BUFFERED_IO=1 RAPID_IO_PROFILE=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" --data "$data" --sizes-json "$sizes" "${strategy[@]}" --steps 110 --display 10 > "$out/${variant}_${run}.log" 2>&1
  fi
 done
done
if [[ $ds == taobao ]]; then torchscript="$root/torchrec_deepfm.py"; else torchscript="$root/torchrec_criteo.py"; fi
for mode in prefetch gpu; do
 for run in 1 2 3; do
  LD_LIBRARY_PATH="$common" OMP_NUM_THREADS=1 /tmp/rapid_torchrec11/bin/torchrun --standalone --nproc-per-node=8 "$torchscript" --data "$stage/torch" --input-mode "$mode" --steps 80 --warmup 30 --out "$out/torch_${mode}_${run}.json" > "$out/torch_${mode}_${run}.log" 2>&1
 done
done
if [[ $ds == kaggle ]]; then vocab=19279324; hetds=criteo_rapid; model=dcnv2; dim=32
elif [[ $ds == terabyte ]]; then vocab=21459313; hetds=criteo_rapid; model=dcnv2; dim=32
else vocab=964982; hetds=taobao; model=dfm; dim=128; fi
for run in 1 2 3; do
 (cd "$root/upstream_fresh/HET-GMP"; RAPID_HET_DATA="$stage/het" RAPID_HET_VOCAB="$vocab" RAPID_HET_BIASES=1 PYTHONPATH="$root/upstream_fresh/HET-GMP/python:/tmp/rapid_het_build/lib" LD_LIBRARY_PATH="/tmp/rapid_het_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common" OMP_NUM_THREADS=1 mpirun -np 8 /tmp/rapid_torchrec11/bin/python -u examples/hetuctr.py --dataset "$hetds" --model "$model" --batch_size 8192 --embed_dim "$dim" --bound 0 --iter 110 --log_every 10 --partition "$stage/het/partition_balanced.npz" --store_rate 1 --benchmark --profile-io-prefix "$out/het_${run}_io" --output "$out/het_${run}.metrics") > "$out/het_${run}.log" 2>&1
done
