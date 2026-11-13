"""Run Herald's LAIA scheduler and Hetu GPU executor on the R1 raw split."""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERALD = Path('/home/longjiu.bty/Rapid/code/upstream_fresh/Herald')
sys.path.insert(0, str(HERALD / 'python'))
import hetu as ht
from hetu import init, laia


def load_raw(path, dense_count, hotness, sizes):
    width = 1 + dense_count + sum(hotness)
    raw = np.memmap(path, dtype=np.int32, mode='r').reshape(-1, width)
    labels = raw[:, :1].astype(np.float32)
    dense = raw[:, 1:1 + dense_count].view(np.float32).copy()
    sparse = raw[:, 1 + dense_count:].copy()
    boundaries = np.cumsum([0] + hotness)
    offsets = np.cumsum([0] + sizes)
    for table in range(len(hotness)):
        field = sparse[:, boundaries[table]:boundaries[table + 1]]
        if field.min() < 0 or field.max() >= sizes[table]:
            raise ValueError(f'sparse IDs exceed table {table} size')
        field += offsets[table]
    return dense, sparse, labels


class Graph:
    def __init__(self):
        self.index = 0

    def variable(self, shape, zero=False, ctx=None):
        self.index += 1
        name = f'r1_param_{self.index}'
        if zero:
            return init.zeros(shape, name=name, ctx=ctx)
        return init.random_normal(
            shape, stddev=float(os.environ.get('RAPID_HERALD_INIT_STD', '0.01')),
            name=name, ctx=ctx)

    def linear(self, x, input_width, output_width, bias=True):
        out = ht.matmul_op(x, self.variable((input_width, output_width)))
        if bias:
            out = out + ht.broadcastto_op(self.variable((output_width,), zero=True), out)
        return out

    def mlp(self, x, input_width, widths, final_relu=False):
        for index, output_width in enumerate(widths):
            x = self.linear(x, input_width, output_width)
            if index + 1 < len(widths) or final_relu:
                x = ht.relu_op(x)
            input_width = output_width
        return x


def build_model(dense, sparse, labels, sizes, hotness, model, dense_count, dim):
    graph = Graph()
    embedding_width = dim + 4 if model in ('wide_deep', 'deepfm') else dim
    weights = graph.variable((sum(sizes), embedding_width), ctx=ht.cpu(0))
    looked_up = ht.embedding_lookup_op(weights, sparse, enable_push_index=True,
                                       ctx=ht.cpu(0))
    boundaries = np.cumsum([0] + hotness)
    pooled = []
    for table, count in enumerate(hotness):
        field = ht.slice_op(looked_up, (0, int(boundaries[table]), 0),
                            (-1, count, embedding_width))
        pooled.append(ht.array_reshape_op(field, (-1, embedding_width)) if count == 1
                      else ht.reduce_sum_op(field, axes=[1]))

    if model == 'dlrm':
        bottom = graph.mlp(dense, dense_count, [512, 256, dim], final_relu=True)
        fields = ht.concatenate_op(
            [ht.array_reshape_op(field, (-1, 1, dim)) for field in [bottom] + pooled], axis=1)
        gram = ht.batch_matmul_op(fields, fields, trans_B=True)
        interactions = [ht.array_reshape_op(ht.slice_op(gram, (0, i, j), (-1, 1, 1)),
                                            (-1, 1))
                        for i in range(1, len(sizes) + 1) for j in range(i)]
        features = ht.concatenate_op([bottom] + interactions, axis=1)
        logits = graph.mlp(features, dim + len(interactions), [1024, 1024, 512, 256, 1])
    elif model == 'dcnv2':
        bottom = graph.mlp(dense, dense_count, [512, 256, 128], final_relu=True)
        width = len(sizes) * dim + 128
        x0 = ht.concatenate_op(pooled + [bottom], axis=1)
        x = x0
        for _ in range(3):
            x = x + ht.mul_op(x0, graph.linear(graph.linear(x, width, 512, bias=False),
                                                512, width))
        logits = graph.mlp(x, width, [1024, 1024, 512, 256, 1])
    elif model == 'dcn':
        width = len(sizes) * dim + dense_count
        x0 = ht.concatenate_op(pooled + [dense], axis=1)
        x = x0
        for _ in range(6):
            cross = ht.mul_op(x0, ht.broadcastto_op(graph.linear(x, width, 1), x0))
            x = x + cross + ht.broadcastto_op(graph.variable((width,), zero=True), x0)
        deep = ht.dropout_op(graph.mlp(x0, width, [1024], final_relu=True), 0.5)
        deep = ht.dropout_op(graph.mlp(deep, 1024, [1024], final_relu=True), 0.5)
        logits = graph.linear(ht.concatenate_op([deep, x], axis=1), 1024 + width, 1)
    else:
        fields = [ht.slice_op(field, (0, 0), (-1, dim)) for field in pooled]
        features = ht.concatenate_op(fields + [dense], axis=1)
        input_width = len(sizes) * dim + dense_count
        widths = [1024, 1024, 1] if model == 'wide_deep' else [400, 400, 400, 1]
        first_order = [ht.slice_op(field, (0, dim), (-1, 1)) for field in pooled]
        logits = graph.mlp(features, input_width, widths) + ht.reduce_sum_op(
            ht.concatenate_op(first_order, axis=1), axes=[1], keepdims=True)
        logits = logits + graph.linear(dense, dense_count, 1)
        if model == 'deepfm':
            stacked = ht.concatenate_op(
                [ht.array_reshape_op(field, (-1, 1, dim)) for field in fields], axis=1)
            field_sum = ht.reduce_sum_op(stacked, axes=[1])
            squared_sum = ht.mul_op(field_sum, field_sum)
            sum_squares = ht.reduce_sum_op(ht.mul_op(stacked, stacked), axes=[1])
            fm = ht.reduce_sum_op((squared_sum + (-1 * sum_squares)) * 0.5,
                                  axes=[1], keepdims=True)
            logits = logits + fm

    prediction = ht.sigmoid_op(logits)
    # Two-class log-sum-exp avoids log(0) when multi-hot interactions saturate sigmoid.
    two_class_logits = ht.concatenate_op([logits * 0, logits], axis=1)
    label_indices = ht.array_reshape_op(labels, (-1,))
    loss = ht.reduce_mean_op(
        ht.softmaxcrossentropy_sparse_op(two_class_logits, label_indices), [0])
    train_op = ht.optim.AdaGradOptimizer(
        learning_rate=float(os.environ.get('RAPID_HERALD_LR', '0.01'))).minimize(loss)
    return loss, prediction, train_op


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--sizes-json', type=Path, required=True)
    parser.add_argument('--dataset', choices=('kaggle', 'terabyte', 'taobao',
                                              't180', 't200', 't510'), required=True)
    parser.add_argument('--model', choices=('dlrm', 'dcnv2', 'dcn', 'wide_deep', 'deepfm'), required=True)
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--steps', type=int, default=510)
    parser.add_argument('--local-batch', type=int, default=8192)
    parser.add_argument('--hotness-json', type=Path)
    parser.add_argument('--local-shared', action='store_true')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    hotness = (json.loads(args.hotness_json.read_text()) if args.hotness_json else
               [1] * 4 if args.dataset == 'taobao' else
               [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1])
    if isinstance(hotness, dict):
        hotness = hotness['lengths']
    dense_count = 1 if args.dataset == 'taobao' else 13
    sizes = json.loads(args.sizes_json.read_text())
    dense, sparse, labels = load_raw(args.data, dense_count, hotness, sizes)
    if len(dense) % (args.local_batch * 8):
        raise ValueError('training split must contain full global batches')
    if shutil.disk_usage(args.out_dir.parent).free < 8 * 1024**3:
        raise OSError('insufficient disk space for Herald logs and reserve')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scheduler = laia.LAIAScheduler(sparse, args.local_batch, dataset='criteo',
                                  local_shared=args.local_shared)
    sparse_input = laia.laia_dataloader_op([[sparse, args.local_batch, 'train']], scheduler, 0,
                                            is_sparse=True)
    labels_input = laia.laia_dataloader_op([[labels, args.local_batch, 'train']], scheduler, 1)
    dense_input = laia.laia_dataloader_op([[dense, args.local_batch, 'train']], scheduler, 2)
    loss, prediction, train_op = build_model(dense_input, sparse_input, labels_input,
                                             sizes, hotness, args.model, dense_count, args.embed_dim)
    epochs = math.ceil(args.steps / (len(dense) // (8 * args.local_batch)))
    executor = ht.Executor({'train': [loss, prediction, labels_input, train_op]},
                           sched=scheduler,
                           sched_params={'dataset_num': 3, 'epoch_num': epochs},
                           dist_strategy=ht.dist.DataParallel(aggregate='Hybrid'),
                           cstable_policy='lru',
                           bsp=int(os.environ.get('RAPID_HERALD_BSP', '-1')),
                           cache_bound=int(os.environ.get('RAPID_HERALD_CACHE_BOUND', '100')),
                           cache_limit=max(1, int(float(os.environ.get('RAPID_HERALD_CACHE_FRACTION', '0.1')) * sum(sizes))),
                           seed=42, log_path=str(args.out_dir))
    start = time.perf_counter()
    last_loss = None
    for step in range(args.steps):
        values = executor.run('train', convert_to_numpy_ret_vals=True)
        last_loss = float(np.asarray(values[0]).mean())
        if not math.isfinite(last_loss):
            raise ValueError('nonfinite Herald training loss')
        if executor.rank == 0 and (step + 1) % 50 == 0:
            print(f'HERALD_PROGRESS {args.dataset} {args.model} '
                  f'{step + 1}/{args.steps} {time.perf_counter() - start:.3f}s',
                  flush=True)
    elapsed = time.perf_counter() - start
    result = {'dataset': args.dataset, 'model': args.model, 'rank': executor.rank,
              'steps': args.steps, 'local_batch': args.local_batch, 'embedding_dim': args.embed_dim,
              'train_seconds': elapsed, 'last_loss': last_loss}
    (args.out_dir / f'rank_{executor.rank}.json').write_text(json.dumps(result) + '\n')
    print('HERALD_R1', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
