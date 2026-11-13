#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

root=/home/longjiu.bty/Rapid/code
out_root=/tmp/rapid_rw_d32_e2e_20261104
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
hlib=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
steps=${RW_STEPS:-510}

for dataset in ${RW_DATASETS:-terabyte kaggle taobao movielens25m_user_temporal}; do
  out=$out_root/$dataset
  mkdir -p "$out"
  case "$dataset" in
    terabyte|kaggle)
      split=/dev/shm/rapid_rw_holdout_20261103/$dataset
      torch_data=/dev/shm/rapid_shuffle2026/$dataset/torch
      native_sizes=/tmp/rapid_rw_e2e_20261103/$dataset/native_sizes_verified.json
      plan=/tmp/rapid_shuffle2026_${dataset}_final_plan
      if [[ $dataset == kaggle ]]; then plan=/tmp/rapid_shuffle2026_kaggle_final_plan_fast; fi
      rapid_args=(--sizes-json "/tmp/rapid_shuffle2026_results/$dataset/plan_sizes.json" --acp-plan "$plan/acp.json")
      dense=13; hot=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1
      train_rows=983040; eval_rows=65536; eval_batches=1; models='dcnv2 dlrm wide_deep deepfm' ;;
    taobao)
      split=/dev/shm/rapid_rw_holdout_20261103/taobao
      torch_data=/dev/shm/rapid_read_fair/taobao/torch
      native_sizes=/tmp/rapid_rw_e2e_20261103/taobao/native_sizes_verified.json
      plan=/tmp/rapid_fresh_stage/taobao_final_plan
      rapid_args=(--plan "$plan" --acp-plan "$plan/acp.json")
      dense=1; hot=1,1,1,1
      train_rows=983040; eval_rows=65536; eval_batches=1; models='dcnv2 dlrm wide_deep deepfm' ;;
    movielens25m_user_temporal)
      split=/dev/shm/rapid_movielens25m_user_temporal_20261103
      torch_data=$split/train
      native_sizes=$split/table_sizes.json
      dense=1; hot=1,1
      train_rows=19922944; eval_rows=5046272; eval_batches=77; models='dlrm neumf' ;;
    *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
  esac
  for model in ${RW_MODELS:-$models}; do
    native_log=$out/hugectr_unique_${model}_b8192_d32_${steps}_adagrad.log
    if ! rg -q '^EVAL_METRICS=' "$native_log" 2>/dev/null; then
      native_data=$split/native_train.bin; native_eval=$split/native_eval.bin
      if [[ $dataset == movielens25m_user_temporal ]]; then
        native_data=$split/train/input.bin; native_eval=$split/eval/input.bin
      fi
      HUGECTR_BUFFERED_IO=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 \
        /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" \
        --data "$native_data" --eval-data "$native_eval" \
        --train-samples "$train_rows" --eval-samples "$eval_rows" \
        --max-eval-batches "$eval_batches" --eval-after-fit \
        --dense-count "$dense" --hotness "$hot" --sizes-json "$native_sizes" --strategy unique \
        --shape-profile reference --embed-dim 32 --model "$model" \
        --batch-size 65536 --steps "$steps" --display 100 --optimizer adagrad --lr 0.01 \
        > "$native_log" 2>&1
    fi
    if [[ $dataset != movielens25m_user_temporal ]]; then
      rapid_log=$out/rapid_${model}_b8192_d32_${steps}_adagrad.log
      if ! rg -q '^EVAL_METRICS=' "$rapid_log" 2>/dev/null; then
        HUGECTR_BUFFERED_IO=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 \
          /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" \
          --data "$split/materialized_train.bin" --eval-data "$split/materialized_eval.bin" \
          --train-samples "$train_rows" --eval-samples "$eval_rows" --eval-after-fit \
          --dense-count "$dense" --hotness "$hot" "${rapid_args[@]}" \
          --shape-profile reference --embed-dim 32 --model "$model" \
          --batch-size 65536 --steps "$steps" --display 100 --optimizer adagrad --lr 0.01 \
          > "$rapid_log" 2>&1
      fi
    fi
    torch_log=$out/torchrec_rowwise_${model}_b8192_d32_${steps}_adagrad.log
    torch_json=${torch_log%.log}.json
    if ! rg -q '"eval_history"' "$torch_json" 2>/dev/null; then
      eval_args=(--eval-last-batch)
      if [[ $dataset == movielens25m_user_temporal ]]; then
        eval_args=(--eval-data "$split/eval" --eval-interval 1000000)
      fi
      TIMEFORMAT='OUTER_PROCESS_SECONDS=%R'
      { time LD_LIBRARY_PATH="$common" OMP_NUM_THREADS=1 \
        /tmp/rapid_torchrec11/bin/torchrun --standalone --nproc-per-node=8 \
        "$root/torchrec_criteo.py" --data "$torch_data" --input-mode prefetch \
        --row-wise-only --shape-profile reference --embed-dim 32 --local-batch 8192 \
        --model "$model" --steps "$steps" --warmup 0 --optimizer adagrad --lr 0.01 \
        "${eval_args[@]}" --dump-plan "$out/torchrec_rowwise_${model}_plan.json" \
        --out "$torch_json"; } > "$torch_log" 2>&1
    fi
    echo "completed $dataset $model steps=$steps" >&2
  done
done
