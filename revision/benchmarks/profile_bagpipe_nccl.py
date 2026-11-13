"""Benchmark BagPipe-sized gradient all-reduces on the configured GPU route."""

import argparse
import os
import time

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--now', type=int, default=800_000)
    parser.add_argument('--later', type=int, default=1_000_000)
    parser.add_argument('--repeat', type=int, default=30)
    args = parser.parse_args()

    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    now = torch.ones((args.now, 32), device='cuda')
    later = torch.ones((args.later, 32), device='cuda')

    def step():
        dist.all_reduce(now)
        dist.all_reduce(later)
        torch.cuda.synchronize()

    for _ in range(5):
        step()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(args.repeat):
        step()
    elapsed = (time.perf_counter() - start) * 1000 / args.repeat
    max_elapsed = torch.tensor([elapsed], device='cuda')
    dist.all_reduce(max_elapsed, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f'now={args.now} later={args.later} max_rank_ms={max_elapsed.item():.3f}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
