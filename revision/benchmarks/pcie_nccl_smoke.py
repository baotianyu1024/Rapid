import os

import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
x = torch.tensor([rank + 1.0], device='cuda')
dist.all_reduce(x)
world = dist.get_world_size()
assert x.item() == world * (world + 1) / 2, (rank, x.item())
print(f'rank={rank} sum={x.item()}', flush=True)
dist.destroy_process_group()
