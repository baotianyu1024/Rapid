"""Time PCIe NCCL old-row movement and same-owner GPU repacking."""

import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist


source = Path(os.environ['RAPID_REMAP_RESULT'])
result = json.loads(source.read_text())
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
device = torch.device('cuda', rank)
dist.init_process_group('nccl')
world = dist.get_world_size()
assert world == 8


def alltoall_ms(counts, repeats=60, vector_bytes=256):
    send_counts = [int(v * vector_bytes) for v in counts[rank]]
    recv_counts = [int(counts[src][rank] * vector_bytes) for src in range(world)]
    send = torch.ones(sum(send_counts), dtype=torch.uint8, device=device)
    recv = torch.empty(sum(recv_counts), dtype=torch.uint8, device=device)
    for _ in range(10):
        dist.all_to_all_single(recv, send, recv_counts, send_counts)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    dist.barrier()
    start.record()
    for _ in range(repeats):
        dist.all_to_all_single(recv, send, recv_counts, send_counts)
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / repeats


counts = result['old_row_cross_gpu_routing_counts']
migration = alltoall_ms(counts)
saved_bytes = int(result['forward_vectors_saved_per_future_step'] * 32 * 4 * 2)
# Keep the same routing proportions for a transfer of the saved per-step bytes.
saved_rows = math.ceil(saved_bytes / 256)
total_rows = sum(map(sum, counts))
scaled = [[round(value * saved_rows / total_rows) for value in row] for row in counts]
saved_transfer = alltoall_ms(scaled)
incremental_routes = result['incremental_forward_routes']
rebuild_routes = result['rebuild_forward_routes']
incremental_forward = alltoall_ms(incremental_routes, vector_bytes=128)
rebuild_forward = alltoall_ms(rebuild_routes, vector_bytes=128)
incremental_backward = alltoall_ms(list(map(list, zip(*incremental_routes))), vector_bytes=128)
rebuild_backward = alltoall_ms(list(map(list, zip(*rebuild_routes))), vector_bytes=128)

repack_total = result['full_rebuild_same_owner_rows_repacked']
local_rows = math.ceil(repack_total / world)
source_rows = torch.randn((local_rows, 64), device=device)
target_rows = torch.empty_like(source_rows)
perm = torch.randperm(local_rows, device=device)
for _ in range(10):
    target_rows.copy_(source_rows.index_select(0, perm))
torch.cuda.synchronize(device)
start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)
dist.barrier()
start.record()
for _ in range(60):
    target_rows.copy_(source_rows.index_select(0, perm))
stop.record()
stop.synchronize()
repack = start.elapsed_time(stop) / 60

local = torch.tensor([migration, saved_transfer, repack, incremental_forward,
                      rebuild_forward, incremental_backward, rebuild_backward], device=device)
dist.all_reduce(local, op=dist.ReduceOp.MAX)
if rank == 0:
    (migration_ms, saved_ms, repack_ms, incremental_forward_ms,
     rebuild_forward_ms, incremental_backward_ms, rebuild_backward_ms) = local.cpu().tolist()
    extra_plan_ms = (result['full_rebuild_seconds'] - result['incremental_update_seconds']) * 1000
    numerator = extra_plan_ms + migration_ms + repack_ms
    paired_step_savings = (incremental_forward_ms + incremental_backward_ms -
                           rebuild_forward_ms - rebuild_backward_ms)
    outcome = dict(nccl_p2p_disabled=os.environ.get('NCCL_P2P_DISABLE'),
                   nccl_ib_disabled=os.environ.get('NCCL_IB_DISABLE'),
                   migration_ms_max_rank=migration_ms,
                   same_owner_gpu_repack_ms_max_rank=repack_ms,
                   saved_payload_equivalent_alltoall_ms_max_rank=saved_ms,
                   incremental_forward_alltoall_ms=incremental_forward_ms,
                   rebuild_forward_alltoall_ms=rebuild_forward_ms,
                   incremental_backward_alltoall_ms=incremental_backward_ms,
                   rebuild_backward_alltoall_ms=rebuild_backward_ms,
                   paired_step_alltoall_savings_ms=paired_step_savings,
                   extra_rebuild_plan_ms=extra_plan_ms,
                   estimated_crossover_steps=math.ceil(numerator / saved_ms),
                   paired_route_crossover_steps=(math.ceil(numerator / paired_step_savings)
                                                 if paired_step_savings > 0 else None),
                   estimate_scope='exact route-count all-to-all microbenchmark and random local gather; not paired training')
    target = source.parent / 'transport_benchmark.json'
    target.write_text(json.dumps(outcome, indent=2) + '\n')
    print(json.dumps(outcome, indent=2), flush=True)
dist.destroy_process_group()
