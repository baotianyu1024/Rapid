"""RAPID HugeCTR DCNv2-style run on a preprocessed Criteo trace."""
import argparse
import ctypes
import json
import os
import shutil
import sys
import time
from pathlib import Path
from argparse import Namespace

process_start = time.perf_counter()
import hugectr
from rapid_acp import read_acp_plan

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--plan', type=Path)
p.add_argument('--sizes-json', type=Path)
p.add_argument('--communication', type=Path)
p.add_argument('--acp-plan', type=Path)
p.add_argument('--strategy', choices=('acp', 'unique', 'reduction', 'mixed'), default='acp')
p.add_argument('--reduction-tables', default='')
p.add_argument('--steps', type=int, default=10)
p.add_argument('--display', type=int, default=1)
p.add_argument('--batch-size', type=int, default=65536)
p.add_argument('--gpus', type=int, default=8, help='Number of local GPUs')
p.add_argument('--embed-dim', type=int, default=32)
p.add_argument('--dense-count', type=int, default=13)
p.add_argument('--hotness', default='3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1')
p.add_argument('--model', choices=('dcnv2', 'dcn', 'dlrm', 'wide_deep', 'deepfm', 'neumf'), default='dcnv2')
p.add_argument('--shape-profile', choices=('legacy', 'reference'), default='legacy')
p.add_argument('--train-samples', type=int, default=1048576)
p.add_argument('--eval-data')
p.add_argument('--eval-samples', type=int, default=65536)
p.add_argument('--max-eval-batches', type=int, default=1)
p.add_argument('--eval-interval', type=int, default=1000000)
p.add_argument('--eval-after-fit', action='store_true', help='Evaluate held-out data after the timed fit')
p.add_argument('--eval-logloss', action='store_true', help='Read evaluation logits and labels to compute exact LogLoss')
p.add_argument('--lr', type=float, default=0.001)
p.add_argument('--optimizer', choices=('sgd', 'adagrad'), default='sgd')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--zero-embedding-init', action='store_true', help='Use identical zero vectors for initialization diagnostics')
p.add_argument('--intra-iteration-overlap', action='store_true', help='Enable HugeCTR embedding/compute stream overlap')
p.add_argument('--inter-iteration-overlap', action='store_true', help='Enable HugeCTR next-batch prefetch overlap')
p.add_argument('--cuda-graph', action='store_true', help='Enable HugeCTR CUDA graph training')
p.add_argument('--data-parallel-tables', default='', help='Comma-separated table IDs to replicate on every GPU')
p.add_argument('--placement-plan', type=Path, help='TorchRec table-wise/data-parallel plan for controlled placement comparison')
p.add_argument('--embark-auto', action='store_true', help='Use the upstream EMBark automatic EBC sharding planner')
p.add_argument('--embark-memory-gb', type=float, default=20.0, help='Per-GPU embedding budget for EMBark planner')
p.add_argument('--initial-snapshot-prefix', type=Path, help='Dump initial dense parameters for paired-run audit')
a = p.parse_args()
if a.eval_after_fit and not a.eval_data:
    p.error('--eval-after-fit requires --eval-data')
if a.gpus < 1 or a.batch_size % a.gpus:
    p.error('gpus must be positive and divide the global batch size')
hot = [int(value) for value in a.hotness.split(',')]
if a.dense_count < 1 or not hot or min(hot) < 1:
    p.error('dense count and all table hotness values must be positive')
if bool(a.plan) == bool(a.sizes_json):
    p.error('provide exactly one of --plan or --sizes-json')
if a.sizes_json:
    sizes = json.loads(a.sizes_json.read_text())
else:
    sizes = [0] * len(hot)
    with (a.plan / 'embedding_mapping.tsv').open() as f:
        for row in f:
            table, _, _, physical = map(int, row.split())
            sizes[table] = max(sizes[table], physical + 1)
if len(sizes) != len(hot) or min(sizes) < 1:
    raise RuntimeError('table sizes must match hotness and contain at least one planned ID')
names = [str(i) for i in range(len(hot))]
dp_ids = [int(x) for x in a.data_parallel_tables.split(',') if x]
if a.placement_plan and dp_ids:
    p.error('use --placement-plan or --data-parallel-tables, not both')
if a.embark_auto and (a.placement_plan or dp_ids or a.strategy != 'acp'):
    p.error('--embark-auto cannot be combined with manual placement or compression strategy')
if len(set(dp_ids)) != len(dp_ids) or any(x < 0 or x >= len(hot) for x in dp_ids):
    p.error('invalid or duplicate data-parallel table')
shard_matrix = [names.copy() for _ in range(a.gpus)]
if a.placement_plan:
    placement = json.loads(a.placement_plan.read_text())
    if set(placement) != {f't{i}' for i in range(len(hot))}:
        p.error('placement plan does not cover all tables')
    shard_matrix = [[] for _ in range(a.gpus)]
    for i in range(len(hot)):
        entry = placement[f't{i}']
        ranks = entry['ranks']
        if entry['sharding_type'] == 'data_parallel':
            if sorted(ranks) != list(range(a.gpus)):
                p.error(f'invalid data-parallel ranks for table {i}')
            dp_ids.append(i)
        elif entry['sharding_type'] == 'row_wise':
            if sorted(ranks) != list(range(a.gpus)):
                p.error(f'invalid row-wise ranks for table {i}')
        elif entry['sharding_type'] != 'table_wise' or len(ranks) != 1:
            p.error(f'unsupported placement for table {i}')
        for rank_id in ranks:
            shard_matrix[rank_id].append(str(i))
mp_names = [n for n in names if int(n) not in dp_ids]
dp_names = [str(i) for i in dp_ids]
if a.embark_auto:
    unique_ids = []
elif a.strategy == 'unique':
    unique_ids = list(range(len(hot)))
elif a.strategy == 'reduction':
    unique_ids = []
elif a.strategy == 'mixed':
    selected = [int(x) for x in a.reduction_tables.split(',') if x]
    if len(set(selected)) != len(selected) or any(x < 0 or x >= len(hot) for x in selected):
        p.error('invalid or duplicate reduction table')
    unique_ids = [i for i in range(len(hot)) if i not in selected]
else:
    if not a.acp_plan and not a.communication and a.plan:
        a.acp_plan = a.plan / 'acp.json'
    if bool(a.communication) == bool(a.acp_plan):
        p.error('ACP requires exactly one of --acp-plan or --communication')
    if a.acp_plan:
        unique_ids = read_acp_plan(a.acp_plan, a.gpus, a.batch_size, hot)
    else:
        unique_ids = json.loads(a.communication.read_text())['acp_unique_tables']
if len(set(unique_ids)) != len(unique_ids) or any(i < 0 or i >= len(hot) for i in unique_ids):
    p.error('invalid ACP Unique table IDs')
unique = [str(i) for i in unique_ids]
reduction = [n for n in names if n not in unique]
if a.embark_auto:
    sys.path.insert(0, str(Path(__file__).resolve().parent /
                           'upstream_fresh/HugeCTR/benchmarks/embedding_collection/hugectr'))
    from sharding import generate_plan
    planner_args = Namespace(optimizer=a.optimizer, dp_threshold=0, dense_threshold=0,
                             sharding_plan='auto', num_gpus_per_node=a.gpus,
                             memory_cap_for_embedding=a.embark_memory_gb,
                             mem_comm_bw_ratio=2000 / 25, mem_comm_work_ratio=8 / 2,
                             dense_comm_work_ratio=4 / 2, batchsize=a.batch_size,
                             use_column_wise_shard=False)
    shard_matrix, shard_strategy, unique, reduction = generate_plan(
        sizes, hot, [a.embed_dim + (4 if a.model in ('wide_deep', 'deepfm')
                                     and a.shape_profile == 'reference' else 0)] * len(hot),
        1, a.gpus, planner_args, False)
    print('EMBARK_PLAN=' + json.dumps(dict(memory_cap_gb=a.embark_memory_gb,
                                           dp_threshold=0, dense_threshold=0,
                                           sharding_plan='auto',
                                           shard_matrix=shard_matrix,
                                           shard_strategy=shard_strategy,
                                           unique=unique, reduction=reduction)), flush=True)
    mp_names = [n[0] if isinstance(n, tuple) else n
                for strategy, group in shard_strategy if strategy == 'mp' for n in group]
    dp_names = [n[0] if isinstance(n, tuple) else n
                for strategy, group in shard_strategy if strategy == 'dp' for n in group]
print(f'COMPRESSION strategy={"embark_auto" if a.embark_auto else a.strategy} Unique={unique} Reduction={reduction}', flush=True)
solver = hugectr.CreateSolver(
    model_name='criteo_'+a.model, seed=a.seed, max_eval_batches=a.max_eval_batches, batchsize=a.batch_size,
    batchsize_eval=a.batch_size, vvgpu=[list(range(a.gpus))], repeat_dataset=True, lr=a.lr,
    metrics_spec={hugectr.MetricsType.AUC: 1.0, hugectr.MetricsType.AverageLoss: 1.0},
    gen_loss_summary=True,
    use_mixed_precision=False, use_cuda_graph=a.cuda_graph, all_reduce_algo=hugectr.AllReduceAlgo.NCCL,
    train_intra_iteration_overlap=a.intra_iteration_overlap,
    train_inter_iteration_overlap=a.inter_iteration_overlap,
    use_embedding_collection=True)
if a.optimizer == 'adagrad':
    optimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.AdaGrad,
                                        update_type=hugectr.Update_t.Global,
                                        initial_accu_value=0.0, epsilon=1e-8)
else:
    optimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.SGD,
                                        update_type=hugectr.Update_t.Local, atomic_update=True)
reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.RawAsync, source=[a.data], eval_source=a.eval_data or a.data,
    check_type=hugectr.Check_t.Non, num_samples=a.train_samples,
    eval_num_samples=a.eval_samples if a.eval_data else a.train_samples,
    slot_size_array=sizes, async_param=hugectr.AsyncParam(num_threads=1,
    num_batches_per_thread=3, shuffle=False, multi_hot_reader=True, is_dense_float=True))
input_setup_start = time.perf_counter()
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(label_dim=1, label_name='label', dense_dim=a.dense_count, dense_name='dense',
    data_reader_sparse_param_array=[hugectr.DataReaderSparseParam('data'+n, h, True, 1)
                                    for n, h in zip(names, hot)]))
print(f'INPUT_SETUP_SECONDS={time.perf_counter() - input_setup_start:.6f}', flush=True)
first_order_reference = a.model in ('wide_deep', 'deepfm') and a.shape_profile == 'reference'
embedding_width = (a.embed_dim + 8 if a.model == 'neumf' else
                   a.embed_dim + 4 if first_order_reference else a.embed_dim)
init_param = hugectr.InitParams(embedding_width, hugectr.Initializer_t.Zero) if a.zero_embedding_init else None
tables = [hugectr.EmbeddingTableConfig(name=n, max_vocabulary_size=size, ev_size=embedding_width,
                                      init_param_or_empty=init_param)
          for n, size in zip(names, sizes)]
ebc = hugectr.EmbeddingCollectionConfig(use_exclusive_keys=True,
                                        comm_strategy=hugectr.CommunicationStrategy.Uniform)
if a.model == 'dlrm':
    for name, table in zip(names, tables):
        ebc.embedding_lookup(table_config=table, bottom_name='data'+name,
                             top_name='emb_vec'+name, combiner='sum')
else:
    ebc.embedding_lookup(table_config=tables, bottom_name=['data'+n for n in names],
                         top_name='embeddings', combiner=['sum'] * len(names))
ebc.shard(shard_matrix=shard_matrix,
          shard_strategy=shard_strategy if a.embark_auto else
                         ([('mp', mp_names), ('dp', dp_names)] if dp_names else [('mp', names)]),
          compression_strategy={hugectr.CompressionStrategy.Unique: [n for n in unique if n in mp_names],
                                hugectr.CompressionStrategy.Reduction: [n for n in reduction if n in mp_names]})
model.add(ebc)
if a.model == 'dlrm':
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Concat,
                                bottom_names=['emb_vec'+n for n in names],
                                top_names=['embeddings']))
if a.model in ('dcnv2', 'dlrm'):
    bottom_width = 128 if a.model == 'dcnv2' else a.embed_dim
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MLP, bottom_names=['dense'],
                                top_names=['dense_mlp'], num_outputs=[512, 256, bottom_width],
                                act_type=hugectr.Activation_t.Relu))
    if a.model == 'dcnv2':
        model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Concat,
                                    bottom_names=['embeddings', 'dense_mlp'], top_names=['concat']))
        model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MultiCross,
                                    bottom_names=['concat'], top_names=['cross'], projection_dim=512,
                                    num_layers=3))
        top_input = 'cross'
        top_outputs = [1024, 1024, 512, 256, 1]
    else:
        model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Interaction,
                                    bottom_names=['dense_mlp', 'embeddings'], top_names=['interaction']))
        top_input = 'interaction'
        top_outputs = ([1024, 1024, 512, 256, 1] if a.shape_profile == 'reference'
                       else [1024, 512, 256, 1])
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MLP, bottom_names=[top_input],
                                top_names=['logit'], num_outputs=top_outputs,
                                activations=[hugectr.Activation_t.Relu] * (len(top_outputs)-1) +
                                [hugectr.Activation_t.Non]))
elif a.model == 'dcn':
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Concat,
                                bottom_names=['embeddings', 'dense'], top_names=['features']))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MultiCross,
                                bottom_names=['features'], top_names=['cross'], num_layers=6))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MLP,
                                bottom_names=['features'], top_names=['deep1'], num_outputs=[1024],
                                activations=[hugectr.Activation_t.Relu]))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Dropout,
                                bottom_names=['deep1'], top_names=['drop1'], dropout_rate=0.5))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MLP,
                                bottom_names=['drop1'], top_names=['deep2'], num_outputs=[1024],
                                activations=[hugectr.Activation_t.Relu]))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Dropout,
                                bottom_names=['deep2'], top_names=['drop2'], dropout_rate=0.5))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Concat,
                                bottom_names=['drop2', 'cross'], top_names=['combined']))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.InnerProduct,
                                bottom_names=['combined'], top_names=['logit'], num_output=1))
elif a.model == 'neumf':
    if len(names) != 2 or hot != [1, 1]:
        p.error('NeuMF requires two single-hot tables')
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Reshape,
                                bottom_names=['embeddings'], top_names=['fields'],
                                leading_dim=embedding_width))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Slice,
                                bottom_names=['fields'], top_names=['mlp_fields', 'gmf_fields'],
                                ranges=[(0, a.embed_dim), (a.embed_dim, embedding_width)]))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Reshape,
                                bottom_names=['mlp_fields'], top_names=['mlp_flat'],
                                leading_dim=2 * a.embed_dim))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Reshape,
                                bottom_names=['gmf_fields'], top_names=['gmf_flat'],
                                leading_dim=16))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Slice,
                                bottom_names=['gmf_flat'], top_names=['user_gmf', 'item_gmf'],
                                ranges=[(0, 8), (8, 16)]))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.ElementwiseMultiply,
                                bottom_names=['user_gmf', 'item_gmf'], top_names=['gmf_product']))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MLP,
                                bottom_names=['mlp_flat'], top_names=['mlp_out'],
                                num_outputs=[256, 256, 128, 64],
                                activations=[hugectr.Activation_t.Relu] * 4))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Concat,
                                bottom_names=['mlp_out', 'gmf_product'], top_names=['neumf_features']))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.InnerProduct,
                                bottom_names=['neumf_features'], top_names=['logit'], num_output=1))
elif first_order_reference:
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Reshape,
                                bottom_names=['embeddings'], top_names=['fields'],
                                leading_dim=embedding_width))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Slice,
                                bottom_names=['fields'], top_names=['vectors', 'linear_terms'],
                                ranges=[(0, a.embed_dim), (a.embed_dim, a.embed_dim + 1)]))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Reshape,
                                bottom_names=['vectors'], top_names=['flat_vectors'],
                                leading_dim=len(names) * a.embed_dim))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Reshape,
                                bottom_names=['linear_terms'], top_names=['flat_linear'],
                                leading_dim=len(names)))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReduceSum,
                                bottom_names=['flat_linear'], top_names=['sparse_linear'], axis=1))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.InnerProduct,
                                bottom_names=['dense'], top_names=['dense_linear'], num_output=1))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Concat,
                                bottom_names=['flat_vectors', 'dense'], top_names=['features']))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MLP, bottom_names=['features'],
                                top_names=['deep_logit'],
                                num_outputs=[400, 400, 400, 1] if a.model == 'deepfm' else [1024, 1024, 1],
                                activations=[hugectr.Activation_t.Relu] * (3 if a.model == 'deepfm' else 2) +
                                [hugectr.Activation_t.Non]))
    if a.model == 'deepfm':
        model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.FmOrder2,
                                    bottom_names=['flat_vectors'], top_names=['fm_vector'], out_dim=a.embed_dim))
        model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReduceSum,
                                    bottom_names=['fm_vector'], top_names=['fm_logit'], axis=1))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Add,
                                bottom_names=['deep_logit', 'sparse_linear', 'dense_linear'] +
                                (['fm_logit'] if a.model == 'deepfm' else []),
                                top_names=['logit']))
else:
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Concat,
                                bottom_names=['embeddings', 'dense'], top_names=['features']))
    deep_outputs = ([1024, 1024, 1] if a.model == 'wide_deep' and a.shape_profile == 'reference'
                    else [1024, 512, 256, 1])
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MLP, bottom_names=['features'],
                                top_names=['deep_logit'], num_outputs=deep_outputs,
                                activations=[hugectr.Activation_t.Relu] * (len(deep_outputs)-1) +
                                [hugectr.Activation_t.Non]))
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.InnerProduct,
                                bottom_names=['features'], top_names=['wide_logit'], num_output=1))
    if a.model == 'deepfm':
        model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.FmOrder2,
                                    bottom_names=['embeddings'], top_names=['fm_vector'], out_dim=a.embed_dim))
        model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReduceSum,
                                    bottom_names=['fm_vector'], top_names=['fm_logit'], axis=1))
        logit_inputs = ['deep_logit', 'wide_logit', 'fm_logit']
    else:
        logit_inputs = ['deep_logit', 'wide_logit']
    model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Add,
                                bottom_names=logit_inputs, top_names=['logit']))
model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names=['logit', 'label'], top_names=['loss']))
compile_start = time.perf_counter()
model.compile()
print(f'MODEL_COMPILE_SECONDS={time.perf_counter() - compile_start:.6f}', flush=True)
if a.initial_snapshot_prefix:
    if shutil.disk_usage(a.initial_snapshot_prefix.parent).free < 2 * 1024**3:
        raise OSError('insufficient space for initial parameter snapshot')
    model.save_params_to_files(str(a.initial_snapshot_prefix), 0)
def read_bytes():
    for line in Path('/proc/self/io').read_text().splitlines():
        if line.startswith('read_bytes:'):
            return int(line.split(':', 1)[1])
    raise RuntimeError('read_bytes unavailable')


before_fit_read = read_bytes()
if os.environ.get('RAPID_NCCL_RESET_AT_FIT') == '1':
    ctypes.CDLL('/tmp/rapid_nccl_payload_trace.so').rapid_nccl_reset()
fit_start = time.perf_counter()
print(f'FIT_START_PROCESS_SECONDS={fit_start - process_start:.6f}', flush=True)
profiler = ctypes.CDLL('libcudart.so') if os.environ.get('RAPID_CUDA_PROFILE') == '1' else None
if profiler and profiler.cudaProfilerStart() != 0:
    raise RuntimeError('cudaProfilerStart failed')
model.fit(max_iter=a.steps, display=a.display, eval_interval=a.eval_interval,
          snapshot=1000000, snapshot_prefix='/tmp/rapid_criteo')
if profiler and profiler.cudaProfilerStop() != 0:
    raise RuntimeError('cudaProfilerStop failed')
print(f'FIT_SECONDS={time.perf_counter() - fit_start:.6f}', flush=True)
print(f'FIT_READ_BYTES={read_bytes() - before_fit_read}', flush=True)
if a.eval_after_fit:
    eval_start = time.perf_counter()
    loss_sum = 0.0
    logit_loss_sum = 0.0
    loss_rows = 0
    eval_scores = []
    eval_labels = []
    for batch in range(a.max_eval_batches):
        if not model.eval():
            raise RuntimeError(f'held-out evaluation ended after {batch} batches')
        if a.eval_logloss:
            import numpy as np
            scores = model.check_out_tensor('logit', hugectr.Tensor_t.Evaluate).reshape(-1)
            labels = model.check_out_tensor('label', hugectr.Tensor_t.Evaluate).reshape(-1)
            count = min(len(scores), len(labels), a.eval_samples - loss_rows)
            scores = scores[:count].astype(np.float64)
            labels = labels[:count].astype(np.float64)
            logit_loss_sum += float(np.logaddexp(0.0, scores).sum() - np.dot(labels, scores))
            print(f'EVAL_SCORE_RANGE={scores.min():.9f},{scores.max():.9f},{scores.mean():.9f}', flush=True)
            # HugeCTR's evaluation BCE layer overwrites its input with sigmoid probabilities.
            probabilities = np.clip(scores, 1e-7, 1 - 1e-7)
            loss_sum -= float(np.sum(labels * np.log(probabilities) +
                                     (1 - labels) * np.log1p(-probabilities)))
            loss_rows += count
            eval_scores.append(scores)
            eval_labels.append(labels)
    print(f'EVAL_METRICS={json.dumps(dict(model.get_eval_metrics()))}', flush=True)
    if a.eval_logloss:
        if loss_rows != a.eval_samples:
            raise RuntimeError(f'LogLoss only covered {loss_rows}/{a.eval_samples} evaluation rows')
        from sklearn.metrics import roc_auc_score
        print(f'EVAL_AUC_EXACT={roc_auc_score(np.concatenate(eval_labels), np.concatenate(eval_scores)):.9f}', flush=True)
        print(f'EVAL_LOGLOSS={loss_sum / loss_rows:.9f}', flush=True)
        print(f'EVAL_LOGLOSS_IF_LOGIT={logit_loss_sum / loss_rows:.9f}', flush=True)
    print(f'EVAL_SECONDS={time.perf_counter() - eval_start:.6f}', flush=True)
print(f'PROCESS_WALL_SECONDS={time.perf_counter() - process_start:.6f}', flush=True)
