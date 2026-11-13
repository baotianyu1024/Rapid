"""TorchRec DCNv2-style benchmark on the same Criteo subset as RAPID."""
import argparse
import json
import os
import time
from pathlib import Path

process_start = time.perf_counter()

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


def auc_from_scores(labels, scores):
    order = np.argsort(scores, kind='stable')
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    positives = int(ordered_labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError('AUC requires both positive and negative validation labels')
    starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    positive_counts = np.add.reduceat(ordered_labels, starts)
    rank_sum = np.sum(positive_counts * (starts + ends + 1) / 2)
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, required=True)
p.add_argument('--steps', type=int, default=80)
p.add_argument('--warmup', type=int, default=10)
p.add_argument('--local-batch', type=int, default=8192)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--lr', type=float, default=0.001)
p.add_argument('--optimizer', choices=('sgd', 'adagrad'), default='sgd')
p.add_argument('--embed-dim', type=int, default=32)
p.add_argument('--model', choices=('dcnv2', 'dcn', 'dlrm', 'wide_deep', 'deepfm', 'neumf'), default='dcnv2')
p.add_argument('--shape-profile', choices=('legacy', 'reference'), default='legacy')
p.add_argument('--planner-hbm-gb', type=float, default=16.0)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--timing-detail', action='store_true', help='synchronize stages for diagnosis; changes timing')
p.add_argument('--audit-embedding-updates', action='store_true')
p.add_argument('--audit-dense-updates', action='store_true')
p.add_argument('--dump-plan', type=Path, help='Write the TorchRec sharding plan on rank 0')
p.add_argument('--local-plan', action='store_true', help='Run deterministic planner on every rank without NCCL object broadcast')
p.add_argument('--no-data-parallel', action='store_true', help='Constrain planner to table and row sharding')
p.add_argument('--row-wise-only', action='store_true', help='Require every embedding table to use row-wise sharding')
p.add_argument('--zero-embedding-init', action='store_true', help='Pair logical embedding initialization with HugeCTR diagnostics')
p.add_argument('--eval-last-batch', action='store_true', help='Hold out the final global batch for AUC and LogLoss')
p.add_argument('--eval-data', type=Path, help='Independent evaluation data in the training ID space')
p.add_argument('--eval-interval', type=int, default=100)
p.add_argument('--input-mode', choices=('mmap', 'ram', 'prepared', 'prefetch', 'gpu'), default='mmap',
               help='prefetch overlaps pinned CPU batches with training; gpu is a diagnostic')
a = p.parse_args()
if a.row_wise_only and a.no_data_parallel:
    p.error('--row-wise-only already excludes data parallel sharding')
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
torch.manual_seed(a.seed)
dist.init_process_group('nccl')
world = dist.get_world_size()
device = torch.device('cuda', rank)
meta = json.loads((a.data / 'input.json').read_text())
dense_count = meta['dense']
offsets = meta['active_id_offsets']
sizes = [v - u for u, v in zip(offsets, offsets[1:])]
hot = meta['lengths']
columns = np.cumsum([0] + hot)
names = [f'f{i}' for i in range(len(sizes))]
width = len(sizes) * a.embed_dim + 128


class DCNV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.ebc = EmbeddingBagCollection(
            tables=[EmbeddingBagConfig(name=f't{i}', embedding_dim=a.embed_dim, num_embeddings=n,
                                       feature_names=[names[i]]) for i, n in enumerate(sizes)],
            device=torch.device('meta'))
        self.bottom = nn.Sequential(nn.Linear(dense_count, 512, device=device), nn.ReLU(),
                                    nn.Linear(512, 256, device=device), nn.ReLU(),
                                    nn.Linear(256, 128, device=device), nn.ReLU())
        self.cross_u = nn.ModuleList([nn.Linear(width, 512, bias=False, device=device) for _ in range(3)])
        self.cross_v = nn.ModuleList([nn.Linear(512, width, device=device) for _ in range(3)])
        self.top = nn.Sequential(nn.Linear(width, 1024, device=device), nn.ReLU(),
                                 nn.Linear(1024, 1024, device=device), nn.ReLU(),
                                 nn.Linear(1024, 512, device=device), nn.ReLU(),
                                 nn.Linear(512, 256, device=device), nn.ReLU(),
                                 nn.Linear(256, 1, device=device))

    def forward(self, dense, sparse, labels, return_logits=False):
        x0 = torch.cat((self.ebc(sparse).values(), self.bottom(dense)), dim=1)
        x = x0
        for u, v in zip(self.cross_u, self.cross_v):
            x = x + x0 * v(u(x))
        logits = self.top(x)
        if return_logits:
            return logits
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)


class DCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ebc = EmbeddingBagCollection(
            tables=[EmbeddingBagConfig(name=f't{i}', embedding_dim=a.embed_dim,
                                       num_embeddings=n, feature_names=[names[i]])
                    for i, n in enumerate(sizes)], device=torch.device('meta'))
        width = len(sizes) * a.embed_dim + dense_count
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width, device=device))
                                         for _ in range(6)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width, device=device))
                                         for _ in range(6)])
        for weight in self.cross_w:
            nn.init.normal_(weight, std=0.01)
        self.deep = nn.Sequential(nn.Linear(width, 1024, device=device), nn.ReLU(),
                                  nn.Dropout(0.5), nn.Linear(1024, 1024, device=device),
                                  nn.ReLU(), nn.Dropout(0.5))
        self.output = nn.Linear(width + 1024, 1, device=device)

    def forward(self, dense, sparse, labels, return_logits=False):
        x0 = torch.cat((self.ebc(sparse).values(), dense), dim=1)
        x = x0
        for weight, bias in zip(self.cross_w, self.cross_b):
            x = x + x0 * (x @ weight).unsqueeze(1) + bias
        logits = self.output(torch.cat((self.deep(x0), x), dim=1))
        if return_logits:
            return logits
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)


class OtherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_order_reference = a.model in ('wide_deep', 'deepfm') and a.shape_profile == 'reference'
        embedding_width = a.embed_dim + 4 if self.first_order_reference else a.embed_dim
        self.ebc = EmbeddingBagCollection(
            tables=[EmbeddingBagConfig(name=f't{i}', embedding_dim=embedding_width, num_embeddings=n,
                                       feature_names=[names[i]]) for i, n in enumerate(sizes)],
            device=torch.device('meta'))
        if a.model == 'dlrm':
            self.bottom = nn.Sequential(nn.Linear(dense_count, 512, device=device), nn.ReLU(),
                                        nn.Linear(512, 256, device=device), nn.ReLU(),
                                        nn.Linear(256, a.embed_dim, device=device), nn.ReLU())
            interactions = (len(sizes) + 1) * len(sizes) // 2
            top_widths = ([1024, 1024, 512, 256, 1] if a.shape_profile == 'reference'
                          else [1024, 512, 256, 1])
            self.top = make_mlp(a.embed_dim + interactions, top_widths)
        else:
            input_width = len(sizes) * a.embed_dim + dense_count
            deep_widths = ([1024, 1024, 1] if a.model == 'wide_deep' else [400, 400, 400, 1])
            if a.shape_profile == 'legacy':
                deep_widths = [1024, 512, 256, 1]
            self.deep = make_mlp(input_width, deep_widths)
            if self.first_order_reference:
                self.dense_linear = nn.Linear(dense_count, 1, device=device)
            else:
                self.wide = nn.Linear(input_width, 1, device=device)

    def forward(self, dense, sparse, labels, return_logits=False):
        embeddings = self.ebc(sparse).values()
        if a.model == 'dlrm':
            bottom = self.bottom(dense)
            fields = torch.cat((bottom.unsqueeze(1),
                                embeddings.reshape(-1, len(sizes), a.embed_dim)), dim=1)
            gram = torch.bmm(fields, fields.transpose(1, 2))
            left, right = torch.tril_indices(len(sizes) + 1, len(sizes) + 1, offset=-1, device=device)
            logits = self.top(torch.cat((bottom, gram[:, left, right]), dim=1))
        else:
            if self.first_order_reference:
                fields_with_linear = embeddings.reshape(-1, len(sizes), a.embed_dim + 4)
                fields = fields_with_linear[:, :, :a.embed_dim]
                features = torch.cat((fields.flatten(1), dense), dim=1)
                logits = self.deep(features) + fields_with_linear[:, :, a.embed_dim].sum(1, keepdim=True)
                logits = logits + self.dense_linear(dense)
            else:
                features = torch.cat((embeddings, dense), dim=1)
                logits = self.deep(features) + self.wide(features)
            if a.model == 'deepfm':
                if not self.first_order_reference:
                    fields = embeddings.reshape(-1, len(sizes), a.embed_dim)
                fm = 0.5 * (fields.sum(dim=1).square() - fields.square().sum(dim=1))
                logits = logits + fm.sum(dim=1, keepdim=True)
        if return_logits:
            return logits
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)


class NeuMF(nn.Module):
    def __init__(self):
        super().__init__()
        if len(sizes) != 2 or hot != [1, 1]:
            raise ValueError('NeuMF requires two single-hot tables')
        self.ebc = EmbeddingBagCollection(
            tables=[EmbeddingBagConfig(name=f't{i}', embedding_dim=a.embed_dim + 8,
                                       num_embeddings=n, feature_names=[names[i]])
                    for i, n in enumerate(sizes)], device=torch.device('meta'))
        layers = []
        input_width = 2 * a.embed_dim
        for output_width in (256, 256, 128, 64):
            layers.extend((nn.Linear(input_width, output_width, device=device), nn.ReLU()))
            input_width = output_width
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(72, 1, device=device)

    def forward(self, dense, sparse, labels, return_logits=False):
        fields = self.ebc(sparse).values().reshape(-1, 2, a.embed_dim + 8)
        mlp = self.mlp(fields[:, :, :a.embed_dim].flatten(1))
        gmf = fields[:, 0, a.embed_dim:] * fields[:, 1, a.embed_dim:]
        logits = self.out(torch.cat((mlp, gmf), dim=1))
        if return_logits:
            return logits
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)


def make_mlp(input_width, outputs):
    layers = []
    for i, output_width in enumerate(outputs):
        layers.append(nn.Linear(input_width, output_width, device=device))
        if i + 1 < len(outputs):
            layers.append(nn.ReLU())
        input_width = output_width
    return nn.Sequential(*layers)


model = (DCNV2() if a.model == 'dcnv2' else DCN() if a.model == 'dcn'
         else NeuMF() if a.model == 'neumf' else OtherModel())
optimizer_class = torch.optim.Adagrad if a.optimizer == 'adagrad' else torch.optim.SGD
optimizer_kwargs = {'lr': a.lr, 'eps': 1e-8} if a.optimizer == 'adagrad' else {'lr': a.lr}
_apply_optimizer_in_backward(optimizer_class, model.ebc.parameters(), optimizer_kwargs.copy())
sharders = [EmbeddingBagCollectionSharder()]
planner = EmbeddingShardingPlanner(
    topology=Topology(world_size=world, local_world_size=world, compute_device='cuda',
                      hbm_cap=int(a.planner_hbm_gb * 2**30),
                      intra_host_bw=14.66833055 * 2**30 / 1000),
    batch_size=a.local_batch,
    constraints={f't{i}': ParameterConstraints(pooling_factors=[float(h)],
                 sharding_types=(['row_wise'] if a.row_wise_only else
                                 ['table_wise', 'row_wise'] if a.no_data_parallel else None))
                 for i, h in enumerate(hot)})
plan = planner.plan(model, sharders) if a.local_plan else planner.collective_plan(model, sharders, dist.group.WORLD)
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
if a.zero_embedding_init:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'ebc' in name:
                param.zero_()


def embedding_sample():
    samples = {}
    for name, value in model.state_dict().items():
        if 'ebc' not in name or 'weight' not in name:
            continue
        if hasattr(value, 'local_shards'):
            tensors = [shard.tensor for shard in value.local_shards()]
        elif isinstance(value, torch.Tensor):
            tensors = [value]
        else:
            continue
        for i, tensor in enumerate(tensors):
            samples[f'{name}:{i}'] = tensor.detach().reshape(-1)[:65536].cpu().clone()
    return samples


before_embeddings = embedding_sample() if a.audit_embedding_updates else None


def dense_sample():
    return {name: param.detach().reshape(-1)[:65536].cpu().clone()
            for name, param in model.named_parameters() if 'ebc' not in name}


before_dense = dense_sample() if a.audit_dense_updates else None
optimizer = CombinedOptimizer([
    KeyedOptimizerWrapper(dict(in_backward_optimizer_filter(model.named_parameters())),
                          lambda params: optimizer_class(params, **optimizer_kwargs)),
    model.fused_optimizer])
input_load_start = time.perf_counter()
sparse_all = np.load(a.data / 'sparse.npy', mmap_mode='r' if a.input_mode == 'mmap' else None)
raw = (np.memmap(a.data / 'input.bin', dtype='<u4', mode='r') if a.input_mode == 'mmap'
       else np.fromfile(a.data / 'input.bin', dtype='<u4')).reshape(-1, 1 + dense_count + sum(hot))
input_load_seconds = time.perf_counter() - input_load_start
num_batches = len(raw) // (world * a.local_batch)
train_batches = num_batches - int(a.eval_last_batch)
if train_batches < 1:
    raise RuntimeError('at least one complete training batch is required')
lengths = torch.cat([torch.full((a.local_batch,), h, dtype=torch.int32, device=device)
                     for h in hot])


def prepare_host(batch, source_raw=raw, source_sparse=sparse_all):
    first = batch * world * a.local_batch + rank * a.local_batch
    sl = slice(first, first + a.local_batch)
    ids = np.asarray(source_sparse[sl], dtype=np.int64)
    values = np.concatenate([(ids[:, columns[t]:columns[t+1]] - offsets[t]).reshape(-1)
                             for t in range(len(hot))])
    values = torch.from_numpy(values)
    dense = torch.from_numpy(source_raw[sl, 1:1 + dense_count].copy().view('<f4'))
    labels = torch.from_numpy(source_raw[sl, 0].astype(np.float32).reshape(-1, 1))
    return values, dense, labels


eval_raw = eval_sparse = None
if a.eval_data:
    eval_meta = json.loads((a.eval_data / 'input.json').read_text())
    if (eval_meta['lengths'] != hot or eval_meta['dense'] != dense_count or
            eval_meta['active_id_offsets'] != offsets):
        raise ValueError('evaluation data must use training table lengths and compact ID space')
    eval_sparse = np.load(a.eval_data / 'sparse.npy', mmap_mode='r')
    eval_raw = np.memmap(a.eval_data / 'input.bin', dtype='<u4', mode='r').reshape(-1, 1 + dense_count + sum(hot))
    if len(eval_raw) % (world * a.local_batch):
        raise ValueError('evaluation rows must divide into global batches')


def evaluate_external(step):
    model.eval()
    local_logits, local_labels = [], []
    with torch.no_grad():
        for batch in range(len(eval_raw) // (world * a.local_batch)):
            values, dense, labels = (x.to(device) for x in prepare_host(batch, eval_raw, eval_sparse))
            kjt = KeyedJaggedTensor.from_lengths_sync(keys=names, values=values, lengths=lengths)
            logits = model(dense, kjt, labels, return_logits=True)
            local_logits.append(logits.float().cpu().reshape(-1).numpy())
            local_labels.append(labels.cpu().reshape(-1).numpy())
    gathered = [None] * world
    dist.all_gather_object(gathered, (np.concatenate(local_logits), np.concatenate(local_labels)))
    result = None
    if rank == 0:
        logits = np.concatenate([x[0] for x in gathered])
        labels = np.concatenate([x[1] for x in gathered])
        terms = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
        result = dict(step=step, auc=auc_from_scores(labels, logits),
                      logloss=float(terms.mean()), rows=len(labels),
                      positives=int(labels.sum()), process_wall_seconds=time.perf_counter()-process_start)
        print('EVAL', json.dumps(result), flush=True)
    model.train()
    return result


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


times, losses, details, eval_history = [], [], [], []
dist.barrier()
torch.cuda.reset_peak_memory_stats(device)
if os.environ.get('RAPID_NCCL_RESET_AT_FIT') == '1':
    import ctypes
    ctypes.CDLL('/tmp/rapid_nccl_payload_trace.so').rapid_nccl_reset()
loop_start = time.perf_counter()
if os.environ.get('RAPID_CUDA_PROFILE') == '1' and torch.cuda.cudart().cudaProfilerStart() != 0:
    raise RuntimeError('cudaProfilerStart failed')
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
    if a.eval_data and (step + 1) % a.eval_interval == 0:
        result = evaluate_external(step + 1)
        if rank == 0:
            eval_history.append(result)
dist.barrier()
loop_wall_seconds = time.perf_counter() - loop_start
if os.environ.get('RAPID_CUDA_PROFILE') == '1' and torch.cuda.cudart().cudaProfilerStop() != 0:
    raise RuntimeError('cudaProfilerStop failed')
io_end = io_snapshot()
eval_metrics = None
if a.eval_data and (a.warmup + a.steps) % a.eval_interval:
    result = evaluate_external(a.warmup + a.steps)
    if rank == 0:
        eval_history.append(result)
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
        all_logits = np.concatenate([x[0] for x in gathered])
        all_labels = np.concatenate([x[1] for x in gathered])
        probs = 1 / (1 + np.exp(-np.clip(all_logits, -40, 40)))
        loss_terms = np.maximum(all_logits, 0) - all_logits * all_labels + np.log1p(np.exp(-np.abs(all_logits)))
        eval_metrics = dict(rows=len(all_labels), auc=auc_from_scores(all_labels, probs),
                            logloss=float(loss_terms.mean()), positives=int(all_labels.sum()))
io_by_rank = [None] * world
dist.all_gather_object(io_by_rank, {k: io_end[k] - io_start[k] for k in io_start})
startup_by_rank = [None] * world
dist.all_gather_object(startup_by_rank, dict(input_load_seconds=input_load_seconds,
                                           input_prepare_seconds=input_prepare_seconds))
all_times, all_memory = [None] * world, [None] * world
dist.all_gather_object(all_times, times)
all_loop_walls = [None] * world
dist.all_gather_object(all_loop_walls, loop_wall_seconds)
dist.all_gather_object(all_memory, torch.cuda.max_memory_allocated(device))
all_details = [None] * world
if a.timing_detail:
    dist.all_gather_object(all_details, details)
audit_results = None
if a.audit_embedding_updates:
    after = embedding_sample()
    local_audit = {name: int(torch.count_nonzero(before != after[name]).item())
                   for name, before in before_embeddings.items() if name in after}
    audit_results = [None] * world
    dist.all_gather_object(audit_results, local_audit)
dense_audit = None
if a.audit_dense_updates:
    after_dense = dense_sample()
    local_dense = {name: int(torch.count_nonzero(before != after_dense[name]).item())
                   for name, before in before_dense.items() if name in after_dense}
    dense_audit = [None] * world
    dist.all_gather_object(dense_audit, local_dense)
if rank == 0:
    step_times = np.max(np.asarray(all_times), axis=0)
    a.out.write_text(json.dumps(dict(method='TorchRec 1.1.0', dataset=meta['dataset'],
        model=a.model, shape_profile=a.shape_profile,
        sharding_mode='row_wise_only' if a.row_wise_only else 'auto',
        embedding_dim=a.embed_dim, seed=a.seed, lr=a.lr, optimizer=a.optimizer,
        ranks=world, local_batch=a.local_batch, warmup=a.warmup,
        steps=a.steps, mean_step_seconds=float(step_times.mean()),
        step_seconds=step_times.tolist(),
        samples_per_second=float(world * a.local_batch / step_times.mean()),
        loop_wall_seconds=max(all_loop_walls),
        loop_samples_per_second=float(world * a.local_batch * (a.warmup + a.steps) / max(all_loop_walls)),
        peak_allocated_bytes_by_rank=all_memory, rank0_loss=losses,
        diagnostic_stage_seconds=(np.max(np.asarray(all_details), axis=0).mean(axis=0).tolist()
                                  if a.timing_detail else None),
        diagnostic_stage_names=(['input_and_zero_grad', 'forward', 'backward', 'optimizer']
                                if a.timing_detail else None),
        input_mode=a.input_mode, io_delta_by_rank=io_by_rank, input_startup_by_rank=startup_by_rank,
        embedding_update_samples=audit_results, dense_update_samples=dense_audit,
        eval_last_batch=eval_metrics,
        eval_history=eval_history,
        scope=('GPU-resident batch replay; excludes preloading' if a.input_mode == 'gpu' else
               'preformatted pinned host batches and async H2D; excludes preparation' if a.input_mode == 'prefetch' else
               'includes input slicing and transfer; excludes offline preprocessing')), indent=2) + '\n')
    print(a.out.read_text(), flush=True)
dist.destroy_process_group()
