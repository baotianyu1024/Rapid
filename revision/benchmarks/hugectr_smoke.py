"""Smoke-test the freshly built HugeCTR against the downloaded Taobao data."""
import argparse
import time
from pathlib import Path

import hugectr
from rapid_acp import read_acp_plan

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--batch-size', type=int, default=8192)
p.add_argument('--steps', type=int, default=10)
p.add_argument('--num-samples', type=int, default=26557961)
p.add_argument('--eval-data')
p.add_argument('--eval-samples', type=int, default=65536)
p.add_argument('--eval-interval', type=int, default=1000000)
p.add_argument('--lr', type=float)
p.add_argument('--display', type=int, default=1)
p.add_argument('--sizes', type=int, nargs=4)
p.add_argument('--hotness', type=int, nargs=4, default=[1, 1, 1, 1])
p.add_argument('--unique', action='store_true')
p.add_argument('--acp-plan', type=Path)
p.add_argument('--reduction-tables', help='comma-separated table IDs for a mixed ACP test')
p.add_argument('--deepfm', action='store_true')
p.add_argument('--embed-dim', type=int, default=128)
p.add_argument('--shape-profile', choices=('legacy', 'reference'), default='legacy')
p.add_argument('--dump-dir', type=Path)
a = p.parse_args()
sizes = a.sizes or [1141729, 846811, 423436, 255875]
names = [str(i) for i in range(len(sizes))]
solver = hugectr.CreateSolver(
    model_name='taobao_smoke', seed=42, max_eval_batches=1,
    batchsize=a.batch_size, batchsize_eval=a.batch_size, vvgpu=[list(range(8))],
    repeat_dataset=True, lr=a.lr if a.lr is not None else (0.001 if a.deepfm else 0.01),
    metrics_spec={hugectr.MetricsType.AUC: 1.0, hugectr.MetricsType.AverageLoss: 0.0},
    gen_loss_summary=True, use_mixed_precision=False,
    use_cuda_graph=False, all_reduce_algo=hugectr.AllReduceAlgo.NCCL,
    use_embedding_collection=True,
)
optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.SGD, update_type=hugectr.Update_t.Local,
    atomic_update=True,
)
reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.RawAsync, source=[a.data],
    eval_source=a.eval_data or a.data, check_type=hugectr.Check_t.Non,
    num_samples=a.num_samples,
    eval_num_samples=a.eval_samples if a.eval_data else a.num_samples, slot_size_array=sizes,
    async_param=hugectr.AsyncParam(num_threads=1, num_batches_per_thread=3,
                                  shuffle=False, multi_hot_reader=True,
                                  is_dense_float=True),
)
input_setup_start = time.perf_counter()
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(
    label_dim=1, label_name='label', dense_dim=1, dense_name='dense',
    data_reader_sparse_param_array=[hugectr.DataReaderSparseParam('data'+n, h, True, 1)
                                    for n, h in zip(names, a.hotness)],
))
print(f'INPUT_SETUP_SECONDS={time.perf_counter() - input_setup_start:.6f}', flush=True)
tables = [hugectr.EmbeddingTableConfig(name=n, max_vocabulary_size=size,
                                       ev_size=a.embed_dim + 4 if a.deepfm else a.embed_dim)
          for n, size in zip(names, sizes)]
ebc = hugectr.EmbeddingCollectionConfig(use_exclusive_keys=True,
                                        comm_strategy=hugectr.CommunicationStrategy.Uniform)
ebc.embedding_lookup(table_config=tables, bottom_name=['data'+n for n in names],
                     top_name='embeddings', combiner=['sum'] * len(names))
shard_args = dict(shard_matrix=[names for _ in range(8)], shard_strategy=[('mp', names)])
if a.acp_plan:
    if a.unique or a.reduction_tables is not None:
        p.error('--acp-plan cannot be combined with explicit strategies')
    unique_ids = read_acp_plan(a.acp_plan, 8, a.batch_size, a.hotness)
    unique = [str(i) for i in unique_ids]
    shard_args['compression_strategy'] = {
        hugectr.CompressionStrategy.Unique: unique,
        hugectr.CompressionStrategy.Reduction: [n for n in names if n not in unique],
    }
    print(f'ACP Unique={unique} Reduction={[n for n in names if n not in unique]}', flush=True)
elif a.reduction_tables is not None:
    reduction = a.reduction_tables.split(',') if a.reduction_tables else []
    if len(set(reduction)) != len(reduction) or any(n not in names for n in reduction):
        p.error('invalid or duplicate reduction table')
    shard_args['compression_strategy'] = {
        hugectr.CompressionStrategy.Unique: [n for n in names if n not in reduction],
        hugectr.CompressionStrategy.Reduction: reduction,
    }
elif a.unique:
    shard_args['compression_strategy'] = {hugectr.CompressionStrategy.Unique: names}
ebc.shard(**shard_args)
model.add(ebc)
if a.deepfm:
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Reshape,
                                bottom_names=['embeddings'], top_names=['per_feature'],
                                leading_dim=a.embed_dim + 4))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Slice,
                                bottom_names=['per_feature'], top_names=['vectors', 'linear_terms'],
                                ranges=[(0, a.embed_dim), (a.embed_dim, a.embed_dim + 1)]))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Reshape,
                                bottom_names=['vectors'], top_names=['flat_vectors'],
                                leading_dim=4 * a.embed_dim))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Reshape,
                                bottom_names=['linear_terms'], top_names=['flat_linear'],
                                leading_dim=4))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReduceSum,
                                bottom_names=['flat_linear'], top_names=['first_order'], axis=1))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.WeightMultiply,
                                bottom_names=['dense'], top_names=['dense_linear'], weight_dims=[1, 1]))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.FmOrder2,
                                bottom_names=['flat_vectors'], top_names=['fm_vectors'], out_dim=a.embed_dim))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReduceSum,
                                bottom_names=['fm_vectors'], top_names=['fm'], axis=1))
    outputs = [400, 400, 400, 1] if a.shape_profile == 'reference' else [256, 256, 1]
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MLP,
                                bottom_names=['flat_vectors'], top_names=['deep'],
                                num_outputs=outputs,
                                activations=[hugectr.Activation_t.Relu] * (len(outputs)-1) +
                                [hugectr.Activation_t.Non]))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Add,
                                bottom_names=['first_order', 'dense_linear', 'fm', 'deep'],
                                top_names=['logit']))
else:
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Concat,
                                bottom_names=['embeddings', 'dense'], top_names=['concat']))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.InnerProduct,
                                bottom_names=['concat'], top_names=['logit'], num_output=1))
model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names=['logit', 'label'], top_names=['loss']))
compile_start = time.perf_counter()
model.compile()
print(f'MODEL_COMPILE_SECONDS={time.perf_counter() - compile_start:.6f}', flush=True)
if a.dump_dir:
    (a.dump_dir / 'before').mkdir(parents=True, exist_ok=True)
    (a.dump_dir / 'after').mkdir(parents=True, exist_ok=True)
    model.embedding_dump(str(a.dump_dir / 'before'))
def read_bytes():
    for line in Path('/proc/self/io').read_text().splitlines():
        if line.startswith('read_bytes:'):
            return int(line.split(':', 1)[1])
    raise RuntimeError('read_bytes unavailable')


fit_read_start = read_bytes()
fit_start = time.perf_counter()
model.fit(max_iter=a.steps, display=a.display, eval_interval=a.eval_interval,
          snapshot=1000000, snapshot_prefix='/tmp/rapid_hugectr_smoke')
print(f'FIT_SECONDS={time.perf_counter() - fit_start:.6f}', flush=True)
print(f'FIT_READ_BYTES={read_bytes() - fit_read_start}', flush=True)
if a.dump_dir:
    model.embedding_dump(str(a.dump_dir / 'after'))
