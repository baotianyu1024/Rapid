#!/usr/bin/env bash
set -euo pipefail

root=/home/longjiu.bty/Rapid
out=/tmp/rapid_reduction_vs_rapid_20261109
mkdir -p "$out"

export HUGECTR_BUFFERED_IO=1
export PYTHONPATH=/tmp/rapid_hugectr_build/lib
export LD_LIBRARY_PATH=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:/usr/local/nvidia/lib64
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8

steps=${RUN_STEPS:-510}
for dataset in ${RUN_DATASETS:-kaggle terabyte}; do
  split=/dev/shm/rapid_rw_holdout_20261103/$dataset
  if [[ $dataset == kaggle ]]; then
    default_model=dcnv2
    plan=/tmp/rapid_shuffle2026_kaggle_final_plan_fast
  elif [[ $dataset == terabyte ]]; then
    default_model=dlrm
    plan=/tmp/rapid_shuffle2026_terabyte_final_plan
  elif [[ $dataset == taobao ]]; then
    default_model=deepfm
    plan=/tmp/rapid_fresh_stage/taobao_final_plan
  else
    echo "Unsupported dataset: $dataset" >&2
    exit 2
  fi
  if [[ $dataset == taobao ]]; then
    dense=1; hotness=1,1,1,1; dim=128; methods=(unique rapid)
  else
    dense=13; hotness=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1
    dim=32; methods=(reduction rapid)
  fi
  model=${RUN_MODEL:-$default_model}
  for method in "${methods[@]}"; do
    log=$out/${dataset}_${model}_${method}_${steps}.log
    if [[ -e $log ]]; then
      echo "Refusing to overwrite $log" >&2
      exit 1
    fi
    if [[ $method != rapid ]]; then
      data=$split/native_train.bin
      eval_data=$split/native_eval.bin
      config=(--sizes-json "/tmp/rapid_rw_e2e_20261103/$dataset/native_sizes_verified.json" --strategy "$method")
    else
      data=$split/materialized_train.bin
      eval_data=$split/materialized_eval.bin
      if [[ $dataset == taobao ]]; then
        config=(--plan "$plan" --acp-plan "$plan/acp.json")
      else
        config=(--sizes-json "/tmp/rapid_shuffle2026_results/$dataset/plan_sizes.json" --acp-plan "$plan/acp.json")
      fi
    fi
    echo "Running $dataset $model $method" >&2
    /tmp/rapid_torchrec11/bin/python -u "$root/code/hugectr_criteo.py" \
      --data "$data" --eval-data "$eval_data" --train-samples 983040 --eval-samples 65536 \
      --dense-count "$dense" --hotness "$hotness" "${config[@]}" \
      --shape-profile reference --embed-dim "$dim" --model "$model" \
      --batch-size 65536 --steps "$steps" --display 100 --optimizer adagrad --lr 0.01 \
      > "$log" 2>&1
    rg '^(FIT_SECONDS|FIT_READ_BYTES|COMPRESSION strategy)=' "$log" || true
  done
done
