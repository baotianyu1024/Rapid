"""Probe full-vocabulary dim32 shard capacity with exact fused AdaGrad, without networking.

This is an embedding allocation/update probe, not an end-to-end benchmark.
Run with torchrun --standalone --nproc-per-node=8 using the TorchRec environment.
"""
import ast
import json
import os
from pathlib import Path

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen, EmbeddingLocation, ComputeDevice,
)

ROOT = Path(__file__).resolve().parents[1]
rank = int(os.environ.get('LOCAL_RANK', 0))
world = int(os.environ.get('WORLD_SIZE', 8))
assert world == 8
torch.cuda.set_device(rank)
generator = ROOT / 'code/upstream_fresh/HugeCTR/benchmarks/embedding_collection/dataset/200table_100B_hotness20.py'
groups = next(ast.literal_eval(node.value) for node in ast.parse(generator.read_text()).body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                 and t.id == 'dataset_info' for t in node.targets))
vocab = [size for count, hotness, size in groups for _ in range(count)]
local_rows = [(n + world - 1 - rank)//world for n in vocab]
torch.cuda.reset_peak_memory_stats()
module = SplitTableBatchedEmbeddingBagsCodegen(
    embedding_specs=[(n, 32, EmbeddingLocation.DEVICE, ComputeDevice.CUDA) for n in local_rows],
    weights_precision=SparseType.FP32, optimizer=OptimType.EXACT_ADAGRAD,
    learning_rate=.01, eps=1e-8, device=torch.device('cuda', rank))
with torch.no_grad():
    for weight in module.split_embedding_weights():
        weight.zero_()
indices = torch.zeros(len(vocab), dtype=torch.int64, device=rank)
offsets = torch.arange(len(vocab)+1, dtype=torch.int64, device=rank)
output = module(indices, offsets)
output.sum().backward()
torch.cuda.synchronize()
for weight in module.split_embedding_weights():
    assert torch.allclose(weight[0], torch.full_like(weight[0], -.01))
state_elements = sum(t.numel() for states in module.split_optimizer_states() for t in states)
weight_elements = sum(w.numel() for w in module.split_embedding_weights())
assert state_elements == weight_elements
result = dict(rank=rank, ranks=world, dataset='T200', tables=len(vocab), dim=32,
              global_vocabulary=sum(vocab), local_vocabulary=sum(local_rows),
              weight_elements=weight_elements, adagrad_state_elements=state_elements,
              max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
              max_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
              first_row_update_verified=True, full_dense_gradient_allocated=False,
              scope='Full vocabulary sharded across 8 GPUs; exact elementwise fused AdaGrad. '
                    'One touched row/table. No dense network, communication, or official input throughput measured.')
out = ROOT / 'response_audit_20261108_recheck' / f't200_fused_capacity_rank{rank}.json'
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
