"""Single-host TorchRec DeepFM run on the freshly prepared Taobao trace."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.optim import _apply_optimizer_in_backward
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig, KeyedJaggedTensor
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology, ParameterConstraints
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, required=True)
p.add_argument('--steps', type=int, default=80)
p.add_argument('--warmup', type=int, default=10)
p.add_argument('--local-batch', type=int, default=8192)
p.add_argument('--embed-dim', type=int, default=128)
p.add_argument('--shape-profile', choices=('legacy', 'reference'), default='legacy')
p.add_argument('--lr', type=float, default=0.001)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--input-mode', choices=('mmap', 'ram', 'prepared', 'prefetch', 'gpu'), default='mmap')
p.add_argument('--timing-detail', action='store_true')
p.add_argument('--dump-plan', type=Path, help='Write the TorchRec sharding plan on rank 0')
p.add_argument('--row-wise-only', action='store_true', help='Require every embedding table to use row-wise sharding')
p.add_argument('--eval-last-batch', action='store_true', help='Hold out the final global batch for AUC and LogLoss')
a = p.parse_args()
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
torch.manual_seed(42)
dist.init_process_group('nccl')
world = dist.get_world_size()
device = torch.device('cuda', rank)
meta = json.loads((a.data / 'input.json').read_text())
offsets = meta['active_id_offsets']
sizes = [v - u for u, v in zip(offsets, offsets[1:])]
names = [f'f{i}' for i in range(len(sizes))]


class DeepFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.ebc = EmbeddingBagCollection(
            tables=[EmbeddingBagConfig(name=f't{i}', embedding_dim=a.embed_dim + 4, num_embeddings=n,
                                       feature_names=[names[i]]) for i, n in enumerate(sizes)],
            device=torch.device('meta'))
        self.dense_linear = nn.Linear(1, 1, bias=False, device=device)
        widths = [400, 400, 400, 1] if a.shape_profile == 'reference' else [256, 256, 1]
        layers = []
        input_width = 4 * a.embed_dim
        for i, output_width in enumerate(widths):
            layers.append(nn.Linear(input_width, output_width, device=device))
            if i + 1 < len(widths):
                layers.append(nn.ReLU())
            input_width = output_width
        self.mlp = nn.Sequential(*layers)

    def forward(self, dense, sparse, labels, return_logits=False):
        z = self.ebc(sparse).values().reshape(-1, 4, a.embed_dim + 4)
        if a.shape_profile == 'reference':
            linear = z[:, :, a.embed_dim].sum(1, keepdim=True) + self.dense_linear(dense)
            v = z[:, :, :a.embed_dim]
        else:
            linear = z[:, :, 0].sum(1, keepdim=True) + self.dense_linear(dense)
            v = z[:, :, 1:a.embed_dim + 1]
        fm = 0.5 * (v.sum(1).square() - v.square().sum(1)).sum(1, keepdim=True)
        logits = linear + fm + self.mlp(v.reshape(-1, 4 * a.embed_dim))
        if return_logits:
            return logits
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


model = DeepFM()
_apply_optimizer_in_backward(torch.optim.SGD, model.ebc.parameters(), {'lr': a.lr})
sharders = [EmbeddingBagCollectionSharder()]
planner = EmbeddingShardingPlanner(
    topology=Topology(world_size=world, local_world_size=world, compute_device='cuda',
                      hbm_cap=16 * 2**30, intra_host_bw=14.66833055 * 2**30 / 1000),
    batch_size=a.local_batch,
    constraints=({f't{i}': ParameterConstraints(sharding_types=['row_wise'])
                  for i in range(len(sizes))} if a.row_wise_only else None))
plan = planner.collective_plan(model, sharders, dist.group.WORLD)
if a.row_wise_only:
    entries = plan.plan['ebc']
    if set(entries) != {f't{i}' for i in range(len(sizes))} or any(
            entry.sharding_type != 'row_wise' or sorted(entry.ranks) != list(range(world))
            for entry in entries.values()):
        raise RuntimeError('planner did not shard every embedding table row-wise across all ranks')
if rank == 0 and a.dump_plan:
    entries = plan.plan['ebc']
    a.dump_plan.write_text(json.dumps({name: dict(sharding_type=entry.sharding_type,
        ranks=entry.ranks, shard_sizes=[] if entry.sharding_spec is None else
        [shard.shard_sizes for shard in entry.sharding_spec.shards])
        for name, entry in entries.items()}, indent=2) + '\n')
model = DistributedModelParallel(module=model, device=device, plan=plan, sharders=sharders)
dense_optimizer = KeyedOptimizerWrapper(
    dict(in_backward_optimizer_filter(model.named_parameters())),
    lambda params: torch.optim.SGD(params, lr=a.lr))
optimizer = CombinedOptimizer([dense_optimizer, model.fused_optimizer])
input_load_start = time.perf_counter()
sparse_all = np.load(a.data / 'sparse.npy', mmap_mode='r' if a.input_mode == 'mmap' else None)
raw = (np.memmap(a.data / 'input.bin', dtype='<u4', mode='r') if a.input_mode == 'mmap'
       else np.fromfile(a.data / 'input.bin', dtype='<u4')).reshape(-1, 6)
input_load_seconds = time.perf_counter() - input_load_start
num_batches = len(raw) // (world * a.local_batch)
if num_batches < 1:
    raise RuntimeError('not enough complete global batches')
train_batches = num_batches - int(a.eval_last_batch)
if train_batches < 1:
    raise RuntimeError('at least one complete training batch is required')
lengths = torch.ones(len(sizes) * a.local_batch, dtype=torch.int32, device=device)


def prepare_host(batch):
    first = batch * world * a.local_batch + rank * a.local_batch
    sl = slice(first, first + a.local_batch)
    ids = np.asarray(sparse_all[sl], dtype=np.int64) - np.asarray(offsets[:-1], dtype=np.int64)
    values = torch.from_numpy(ids.T.copy().reshape(-1))
    dense = torch.from_numpy(raw[sl, 1].copy().view('<f4').reshape(-1, 1))
    labels = torch.from_numpy(raw[sl, 0].astype(np.float32).reshape(-1, 1))
    return values, dense, labels


gpu_batches = None
host_batches = None
input_prepare_start = time.perf_counter()
if a.input_mode in ('prepared', 'prefetch'):
    host_batches = [tuple(x.pin_memory() for x in prepare_host(b)) for b in range(num_batches)]
if a.input_mode == 'gpu':
    gpu_batches = [tuple(x.to(device) for x in prepare_host(b)) for b in range(num_batches)]
    torch.cuda.synchronize(device)
copy_stream = torch.cuda.Stream(device=device) if a.input_mode == 'prefetch' else None


def prefetch(batch):
    with torch.cuda.stream(copy_stream):
        return tuple(x.to(device, non_blocking=True) for x in host_batches[batch])


next_gpu_batch = prefetch(0) if copy_stream is not None else None
input_prepare_seconds = time.perf_counter() - input_prepare_start


def io_snapshot():
    return {k: int(v) for k, v in (line.split(':', 1) for line in
            Path('/proc/self/io').read_text().splitlines())}


times, losses, details = [], [], []
dist.barrier()
torch.cuda.reset_peak_memory_stats(device)
loop_start = time.perf_counter()
for step in range(a.warmup + a.steps):
    if step == a.warmup:
        io_start = io_snapshot()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    batch = step % train_batches
    if copy_stream is not None:
        torch.cuda.current_stream(device).wait_stream(copy_stream)
        values, dense, labels = next_gpu_batch
        for tensor in next_gpu_batch:
            tensor.record_stream(torch.cuda.current_stream(device))
        next_gpu_batch = prefetch((batch + 1) % train_batches)
    elif gpu_batches is not None:
        values, dense, labels = gpu_batches[batch]
    elif host_batches is not None:
        values, dense, labels = (x.to(device, non_blocking=True) for x in host_batches[batch])
    else:
        values, dense, labels = (x.to(device) for x in prepare_host(batch))
    kjt = KeyedJaggedTensor.from_lengths_sync(keys=names, values=values, lengths=lengths)
    optimizer.zero_grad()
    if a.timing_detail:
        torch.cuda.synchronize(device)
        after_input = time.perf_counter()
    loss = model(dense, kjt, labels)
    if a.timing_detail:
        torch.cuda.synchronize(device)
        after_forward = time.perf_counter()
    loss.backward()
    if a.timing_detail:
        torch.cuda.synchronize(device)
        after_backward = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize(device)
    end = time.perf_counter()
    elapsed = end - start
    if step >= a.warmup:
        times.append(elapsed)
        losses.append(loss.item())
        if a.timing_detail:
            details.append([after_input-start, after_forward-after_input,
                            after_backward-after_forward, end-after_backward])
dist.barrier()
loop_wall = time.perf_counter() - loop_start
io_end = io_snapshot()
eval_metrics = None
if a.eval_last_batch:
    model.eval()
    with torch.no_grad():
        values, dense, labels = (x.to(device) for x in prepare_host(num_batches - 1))
        kjt = KeyedJaggedTensor.from_lengths_sync(keys=names, values=values, lengths=lengths)
        logits = model(dense, kjt, labels, return_logits=True)
        local_eval = (logits.float().cpu().reshape(-1).numpy(), labels.cpu().reshape(-1).numpy())
    gathered = [None] * world
    dist.all_gather_object(gathered, local_eval)
    if rank == 0:
        from sklearn.metrics import roc_auc_score
        all_logits = np.concatenate([x[0] for x in gathered])
        all_labels = np.concatenate([x[1] for x in gathered])
        probs = 1 / (1 + np.exp(-np.clip(all_logits, -40, 40)))
        loss_terms = np.maximum(all_logits, 0) - all_logits * all_labels + np.log1p(np.exp(-np.abs(all_logits)))
        eval_metrics = dict(rows=len(all_labels), auc=float(roc_auc_score(all_labels, probs)),
                            logloss=float(loss_terms.mean()), positives=int(all_labels.sum()))
io_by_rank = [None] * world
dist.all_gather_object(io_by_rank, {k: io_end[k] - io_start[k] for k in io_start})
startup_by_rank = [None] * world
dist.all_gather_object(startup_by_rank, dict(input_load_seconds=input_load_seconds,
                                           input_prepare_seconds=input_prepare_seconds))
all_times = [None] * world
all_memory = [None] * world
all_loop_walls = [None] * world
dist.all_gather_object(all_times, times)
dist.all_gather_object(all_memory, torch.cuda.max_memory_allocated(device))
dist.all_gather_object(all_loop_walls, loop_wall)
all_details = [None] * world
if a.timing_detail:
    dist.all_gather_object(all_details, details)
if rank == 0:
    step_times = np.max(np.asarray(all_times), axis=0)
    a.out.write_text(json.dumps(dict(
        method='TorchRec 1.1.0', dataset='Taobao subset', model='DeepFM',
        sharding_mode='row_wise_only' if a.row_wise_only else 'auto',
        embedding_width=a.embed_dim + 4, used_embedding_width=a.embed_dim + 1,
        shape_profile=a.shape_profile,
        ranks=world, local_batch=a.local_batch, warmup=a.warmup, steps=a.steps, lr=a.lr,
        mean_step_seconds=float(step_times.mean()),
        step_seconds=step_times.tolist(), input_mode=a.input_mode, io_delta_by_rank=io_by_rank,
        input_startup_by_rank=startup_by_rank,
        samples_per_second=float(world * a.local_batch / step_times.mean()),
        loop_wall_seconds=max(all_loop_walls),
        loop_samples_per_second=float(world * a.local_batch * (a.warmup + a.steps) / max(all_loop_walls)),
        peak_allocated_bytes_by_rank=all_memory, rank0_loss=losses, eval_last_batch=eval_metrics,
        diagnostic_stage_seconds=(np.max(np.asarray(all_details), axis=0).mean(axis=0).tolist()
                                  if a.timing_detail else None),
        diagnostic_stage_names=(['input_and_zero_grad', 'forward', 'backward', 'optimizer']
                                if a.timing_detail else None),
        scope=('GPU-resident batch replay; excludes preloading' if a.input_mode == 'gpu' else
               'preformatted pinned host batches and async H2D; excludes preparation' if a.input_mode == 'prefetch' else
               'includes input slicing and transfer; excludes offline preprocessing'),
    ), indent=2) + '\n')
    print(a.out.read_text(), flush=True)
dist.destroy_process_group()
