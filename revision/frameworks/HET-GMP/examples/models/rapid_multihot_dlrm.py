"""Multi-hot DLRM using HET-GMP's shared embedding table."""
import os

import hetu as ht
from hetu import init
from hetu.gpu_ops.SharedTable import SharedTableOp


HOT = [int(value) for value in os.environ['RAPID_HET_HOTNESS'].split(',')]


def concat_tree(items):
    while len(items) > 1:
        items = [ht.concat_op(items[index], items[index + 1], axis=1)
                 if index + 1 < len(items) else items[index]
                 for index in range(0, len(items), 2)]
    return items[0]


def DLRM_MULTI(name, dense, sparse, labels, embed_dim, rank, nrank,
               device_id, bound, root, storage):
    vocab = int(os.environ['RAPID_HET_VOCAB'])
    dense_width = int(os.environ.get('RAPID_HET_DENSE_DIM', '13'))
    lr = float(os.environ.get('RAPID_HET_LR', '0.01'))
    if nrank == 1:
        table = init.random_normal((vocab, embed_dim), stddev=0.01, ctx=ht.gpu(device_id))
        features = ht.embedding_lookup_op(table, sparse, ctx=ht.gpu(device_id))
    else:
        features = SharedTableOp(sparse, rank, nrank, (vocab, embed_dim), lr,
                                 bounds=(bound, bound), root_arr=root, storage_arr=storage,
                                 ctx=ht.gpu(device_id))
    pooled = []
    start = 0
    for hot in HOT:
        field = ht.slice_op(features, (0, start, 0), (-1, hot, embed_dim))
        pooled.append(ht.array_reshape_op(field, (-1, embed_dim)) if hot == 1
                      else ht.reduce_sum_op(field, axes=1))
        start += hot

    x = dense
    for index, (input_width, output_width) in enumerate(
            ((dense_width, 512), (512, 256), (256, embed_dim))):
        weight = init.random_normal((input_width, output_width), stddev=0.01,
                                    name=f'dlrm_bottom{index}_weight')
        bias = init.zeros((output_width,), name=f'dlrm_bottom{index}_bias')
        result = ht.matmul_op(x, weight)
        x = ht.relu_op(result + ht.broadcastto_op(bias, result))

    fields = [x] + pooled
    pairwise = [ht.reduce_sum_op(ht.mul_op(fields[row], fields[col]), axes=1, keepdims=True)
                for row in range(len(fields)) for col in range(row + 1, len(fields))]
    x = concat_tree([x, concat_tree(pairwise)])
    width = embed_dim + len(pairwise)
    for index, output_width in enumerate((1024, 1024, 512, 256, 1)):
        weight = init.random_normal((width, output_width), stddev=0.01,
                                    name=f'dlrm_top{index}_weight')
        bias = init.zeros((output_width,), name=f'dlrm_top{index}_bias')
        result = ht.matmul_op(x, weight)
        result = result + ht.broadcastto_op(bias, result)
        x = ht.relu_op(result) if output_width != 1 else result
        width = output_width
    prediction = ht.sigmoid_op(x)
    loss = ht.reduce_mean_op(ht.binarycrossentropy_op(prediction, labels), [0])
    train = ht.optim.SGDOptimizer(learning_rate=lr).minimize(loss)
    return loss, prediction, labels, train
