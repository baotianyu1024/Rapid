#!/usr/bin/env bash
set -euo pipefail

root=/home/longjiu.bty/Rapid
out=/tmp/rapid_paper_model_profiles_20261109
mkdir -p "$out"

export HUGECTR_BUFFERED_IO=1
export PYTHONPATH=/tmp/rapid_hugectr_build/lib
export LD_LIBRARY_PATH=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8 RAPID_CUDA_PROFILE=1

for dataset in ${PROFILE_DATASETS:-kaggle taobao}; do
  split=/dev/shm/rapid_rw_holdout_20261103/$dataset
  if [[ $dataset == kaggle ]]; then
    model=dcn; dense=13; dim=32; baseline=reduction
    hotness=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1
    plan=/tmp/rapid_shuffle2026_kaggle_final_plan_fast
    rapid_config=(--sizes-json /tmp/rapid_shuffle2026_results/kaggle/plan_sizes.json --acp-plan "$plan/acp.json")
  elif [[ $dataset == taobao ]]; then
    model=deepfm; dense=1; dim=128; baseline=unique; hotness=1,1,1,1
    plan=/tmp/rapid_fresh_stage/taobao_final_plan
    rapid_config=(--plan "$plan" --acp-plan "$plan/acp.json")
  else
    echo "Unsupported dataset: $dataset" >&2
    exit 2
  fi
  for method in "$baseline" rapid; do
    name=${dataset}_${model}_${method}
    if [[ -e $out/$name.nsys-rep || -e $out/$name.log ]]; then
      echo "Refusing to overwrite $name" >&2
      exit 1
    fi
    if [[ $method == rapid ]]; then
      data=$split/materialized_train.bin
      config=("${rapid_config[@]}")
    else
      data=$split/native_train.bin
      config=(--sizes-json "/tmp/rapid_rw_e2e_20261103/$dataset/native_sizes_verified.json" --strategy "$baseline")
    fi
    echo "Profiling $name" >&2
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda,nvtx --sample=none -o "$out/$name" \
      /tmp/rapid_torchrec11/bin/python -u "$root/code/hugectr_criteo.py" \
      --data "$data" --train-samples 983040 --dense-count "$dense" --hotness "$hotness" \
      "${config[@]}" --shape-profile reference --embed-dim "$dim" --model "$model" \
      --batch-size 65536 --steps 40 --display 20 --optimizer adagrad --lr 0.01 \
      > "$out/$name.log" 2>&1
  done
done
