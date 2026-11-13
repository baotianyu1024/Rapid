"""Minimal NCCL collective smoke test for the V100 environment."""
import os
import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
x = torch.ones(1, device='cuda') * (rank + 1)
dist.all_reduce(x)
assert x.item() == sum(range(1, dist.get_world_size() + 1))
print(rank, x.item(), flush=True)
dist.destroy_process_group()
