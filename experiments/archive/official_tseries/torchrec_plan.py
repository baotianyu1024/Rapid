"""Run the unmodified upstream planner on meta tensors; no training/GPU execution."""
import argparse
import ast
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path('/mnt/hugui/longjiu.bty')
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2] / 'sources/torchrec-v1.1.0')
parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'plan.json')
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--uniform-dim', type=int)
parser.add_argument('--hbm-gib', type=int, default=32)
parser.add_argument('--intra-gib-s', type=float, default=600)
args = parser.parse_args()
if args.out.exists():
 print("Existing completed official plan",args.out);sys.exit(0)
sys.path.insert(0, str(args.source))
import torch
from torch import nn
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology, ParameterConstraints

logging.basicConfig(level=logging.INFO)
meta = json.loads((args.data / 'input.json').read_text())
sizes = [b-a for a,b in zip(meta['active_id_offsets'], meta['active_id_offsets'][1:])]
lengths = meta['lengths']; dims=([args.uniform_dim]*len(lengths) if args.uniform_dim else meta['dimensions'])
dim, p, b = args.dim, meta['ranks'], meta['local_batch']
assert len(sizes) == len(lengths)

def mlp_params(widths):
    return sum((a + 1) * z for a, z in zip(widths, widths[1:]))

# RAPID's bottom MLP ends at 128; concatenation includes the 26 embeddings.
interaction = sum(dims) + 128
dense_params = (mlp_params([meta['dense'], 512, 256, 128])
                + 3 * (2 * interaction * 512 + interaction)
                + mlp_params([interaction, 1024, 1024, 512, 256, 1]))
model = nn.Module()
model.ebc = EmbeddingBagCollection(tables=[EmbeddingBagConfig(
    name=f't{i}', embedding_dim=dims[i], num_embeddings=n,
    feature_names=[f'f{i}'], pooling=PoolingType.SUM)
    for i, n in enumerate(sizes)], device=torch.device('meta'))
# Dense parameter proxy is sufficient for upstream storage reservation; not a forward model.
model.dense = nn.Parameter(torch.empty(dense_params, device='meta'))
for param in model.ebc.parameters():
    param._optimizer_classes = [torch.optim.SGD]
topology = Topology(world_size=p, local_world_size=p, compute_device='cuda',
                    hbm_cap=args.hbm_gib * 2**30,
                    intra_host_bw=args.intra_gib_s * 2**30 / 1000)
constraints = {f't{i}': ParameterConstraints(pooling_factors=[float(k)]) for i, k in enumerate(lengths)}
planner = EmbeddingShardingPlanner(topology=topology, batch_size=b, constraints=constraints)
plan = planner.plan(model, [EmbeddingBagCollectionSharder()])
tables = []
for i, n in enumerate(sizes):
    entry = plan.plan['ebc'][f't{i}']
    tables.append(dict(table=i, rows=n, pooling=lengths[i], dim=dims[i],
                       sharding_type=entry.sharding_type, kernel=entry.compute_kernel,
                       ranks=entry.ranks, shards=[] if entry.sharding_spec is None else [
                           dict(size=s.shard_sizes, offset=s.shard_offsets, rank=s.placement.rank())
                           for s in entry.sharding_spec.shards]))
result = dict(source=str(args.source), commit=subprocess.check_output(
    ['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    torch_version=torch.__version__, ranks=p, local_batch=b, dim=(args.uniform_dim or "per-table official"), dimensions=dims,
    hbm_gib=args.hbm_gib, intra_gib_s=args.intra_gib_s,
    topology='single host, remaining Topology defaults',
    storage_reservation='upstream heuristic, 15% + dense*6 + KJT*20',
    optimizer='SGD without momentum', dense_parameters=dense_params,
    num_proposals=planner._num_proposals, num_plans=planner._num_plans,
    tables=tables)
args.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({t['table']: (t['sharding_type'], t['ranks']) for t in tables}, indent=2))
