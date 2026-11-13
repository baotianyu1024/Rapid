#!/usr/bin/env bash
set -u

root=/home/longjiu.bty/Rapid/code
split=/dev/shm/rapid_rw_holdout_20261103/taobao
torch_data=/tmp/rapid_fresh_stage/taobao_8_v2/data
plan=/tmp/rapid_fresh_stage/taobao_final_plan
out=/tmp/rapid_rw_e2e_20261103/taobao
mkdir -p "$out"
common=/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64:/usr/local/nvidia/lib64
hlib=/tmp/rapid_hugectr_build/lib:/tmp/rapid_cuda12/lib:/tmp/rapid_het_deps/lib:/tmp/rapid_deps/lib:/tmp/rapid_deps/lib64:/tmp/rapid_cudnn8/nvidia/cudnn/lib:$common
export LD_PRELOAD=/tmp/rapid_nccl_new/nvidia/nccl/lib/libnccl.so.2
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN

for model in dcnv2 dlrm wide_deep deepfm; do
  rapid_log=$out/rapid_${model}_b8192_d64_510.log
  torch_log=$out/torchrec_rowwise_${model}_b8192_d64_510.log
  torch_json=$out/torchrec_rowwise_${model}_b8192_d64_510.json
  if ! rg -q '^EVAL_METRICS=' "$rapid_log" 2>/dev/null; then
    HUGECTR_BUFFERED_IO=1 PYTHONPATH=/tmp/rapid_hugectr_build/lib LD_LIBRARY_PATH="$hlib" OMP_NUM_THREADS=8 \
      /tmp/rapid_torchrec11/bin/python -u "$root/hugectr_criteo.py" \
      --data "$split/materialized_train.bin" --eval-data "$split/materialized_eval.bin" \
      --train-samples 983040 --eval-samples 65536 --eval-after-fit \
      --dense-count 1 --hotness 1,1,1,1 --plan "$plan" --acp-plan "$plan/acp.json" \
      --shape-profile reference --embed-dim 64 --model "$model" \
      --batch-size 65536 --steps 510 --display 100 \
      > "$rapid_log" 2>&1 || echo "FAILED rapid taobao $model" >&2
  fi
  if ! rg -q '"eval_last_batch"' "$torch_json" 2>/dev/null; then
    TIMEFORMAT='OUTER_PROCESS_SECONDS=%R'
    { time LD_LIBRARY_PATH="$common" OMP_NUM_THREADS=1 \
      /tmp/rapid_torchrec11/bin/torchrun --standalone --nproc-per-node=8 \
      "$root/torchrec_criteo.py" --data "$torch_data" --input-mode prefetch \
      --row-wise-only --shape-profile reference --embed-dim 64 --local-batch 8192 \
      --model "$model" --steps 510 --warmup 0 --eval-last-batch \
      --dump-plan "$out/torchrec_rowwise_${model}_plan.json" --out "$torch_json"; } \
      > "$torch_log" 2>&1 || echo "FAILED torchrec taobao $model" >&2
  fi
done
