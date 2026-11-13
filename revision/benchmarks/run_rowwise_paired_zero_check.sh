#!/usr/bin/env bash
set -u

root=/home/longjiu.bty/Rapid/code
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
hlib=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN

for dataset in ${RW_DATASETS:-terabyte kaggle}; do
  split=/dev/shm/rapid_rw_holdout_20261103/$dataset
  out=/tmp/rapid_rw_e2e_20261103/$dataset
  case "$dataset" in
    kaggle) plan=/tmp/rapid_shuffle2026_kaggle_final_plan_fast
      native_sizes=$out/native_sizes_verified.json ;;
    terabyte) plan=/tmp/rapid_shuffle2026_terabyte_final_plan
      native_sizes=$out/native_sizes_verified.json ;;
  esac
  for model in ${RW_MODELS:-dcnv2 dlrm wide_deep deepfm}; do
    for method in rapid native; do
      if [[ $method == rapid ]]; then
        data=$split/materialized_train.bin; eval_data=$split/materialized_eval.bin
        sizes=(--sizes-json "/tmp/rapid_shuffle2026_results/$dataset/plan_sizes.json")
        strategy=(--acp-plan "$plan/acp.json")
      else
        data=$split/native_train.bin; eval_data=$split/native_eval.bin
        sizes=(--sizes-json "$native_sizes")
        strategy=(--strategy unique)
      fi
      log=$out/${method}_${model}_paired_zero_b8192_d64_510.log
      if rg -q '^EVAL_METRICS=' "$log" 2>/dev/null; then
        continue
      fi
      HUGECTR_BUFFERED_IO=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 \
        /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" \
        --data "$data" --eval-data "$eval_data" --train-samples 983040 \
        --eval-samples 65536 --eval-after-fit "${sizes[@]}" "${strategy[@]}" \
        --shape-profile reference --embed-dim 64 --model "$model" \
        --batch-size 65536 --steps 510 --display 100 --zero-embedding-init \
        > "$log" 2>&1 || echo "FAILED zero $dataset $model $method" >&2
    done
  done
done
