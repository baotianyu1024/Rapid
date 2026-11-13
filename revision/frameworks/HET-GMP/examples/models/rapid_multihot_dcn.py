"""DCNv2-style model using HET-GMP's native shared embedding table."""
import os

import hetu as ht
from hetu import init
from hetu.gpu_ops.SharedTable import SharedTableOp

HOT = [int(value) for value in os.environ.get(
    'RAPID_HET_HOTNESS', '3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1').split(',')]


def DCNV2(name, dense, sparse, labels, embed_dim, rank, nrank, device_id, bound, root, storage):
    vocab = int(os.environ['RAPID_HET_VOCAB'])
    dense_dim = int(os.environ.get('RAPID_HET_DENSE_DIM', '13'))
    learning_rate = float(os.environ.get('RAPID_HET_LR', '0.001'))
    if nrank == 1:
        table = init.random_normal((vocab, embed_dim), stddev=0.01, ctx=ht.gpu(device_id))
        features = ht.embedding_lookup_op(table, sparse, ctx=ht.gpu(device_id))
    else:
        features = SharedTableOp(sparse, rank, nrank, (vocab, embed_dim), learning_rate,
                                 bounds=(bound, bound), root_arr=root, storage_arr=storage,
                                 ctx=ht.gpu(device_id))
    pooled = []
    start = 0
    for hot in HOT:
        values = ht.slice_op(features, (0, start, 0), (-1, hot, embed_dim))
        pooled.append(ht.array_reshape_op(values, (-1, embed_dim)) if hot == 1
                      else ht.reduce_sum_op(values, axes=1))
        start += hot
    emb = pooled[0]
    for part in pooled[1:]:
        emb = ht.concat_op(emb, part, axis=1)

    def linear(x, n_in, n_out, name):
        weight = init.random_normal((n_in, n_out), stddev=0.01, name=name+'_weight')
        bias = init.zeros((n_out,), name=name+'_bias')
        result = ht.matmul_op(x, weight)
        return result + ht.broadcastto_op(bias, result)

    x = dense
    for i, (n_in, n_out) in enumerate([(dense_dim, 512), (512, 256), (256, 128)]):
        x = ht.relu_op(linear(x, n_in, n_out, f'bottom{i}'))
    x0 = ht.concat_op(emb, x, axis=1)
    x = x0
    width = len(HOT) * embed_dim + 128
    for i in range(3):
        u = init.random_normal((width, 512), stddev=0.01, name=f'cross{i}_u')
        v = init.random_normal((512, width), stddev=0.01, name=f'cross{i}_v')
        bias = init.zeros((width,), name=f'cross{i}_bias')
        product = ht.matmul_op(ht.matmul_op(x, u), v)
        product = product + ht.broadcastto_op(bias, product)
        x = x + ht.mul_op(x0, product)
    for i, (n_in, n_out) in enumerate([(width, 1024), (1024, 1024), (1024, 512),
                                       (512, 256), (256, 1)]):
        x = linear(x, n_in, n_out, f'top{i}')
        if i != 4:
            x = ht.relu_op(x)
    prediction = ht.sigmoid_op(x)
    loss = ht.reduce_mean_op(ht.binarycrossentropy_op(prediction, labels), [0])
    train = ht.optim.SGDOptimizer(learning_rate=learning_rate).minimize(loss)
    return loss, prediction, labels, train


def DCN(name, dense, sparse, labels, embed_dim, rank, nrank, device_id, bound, root, storage):
    vocab = int(os.environ['RAPID_HET_VOCAB'])
    dense_dim = int(os.environ.get('RAPID_HET_DENSE_DIM', '13'))
    learning_rate = float(os.environ.get('RAPID_HET_LR', '0.01'))
    if nrank == 1:
        table = init.random_normal((vocab, embed_dim), stddev=0.01, ctx=ht.gpu(device_id))
        features = ht.embedding_lookup_op(table, sparse, ctx=ht.gpu(device_id))
    else:
        features = SharedTableOp(sparse, rank, nrank, (vocab, embed_dim), learning_rate,
                                 bounds=(bound, bound), root_arr=root, storage_arr=storage,
                                 ctx=ht.gpu(device_id))
    pooled = []
    start = 0
    for hot in HOT:
        values = ht.slice_op(features, (0, start, 0), (-1, hot, embed_dim))
        pooled.append(ht.array_reshape_op(values, (-1, embed_dim)) if hot == 1
                      else ht.reduce_sum_op(values, axes=1))
        start += hot
    emb = pooled[0]
    for part in pooled[1:]:
        emb = ht.concat_op(emb, part, axis=1)

    width = len(HOT) * embed_dim + dense_dim
    x0 = ht.concat_op(emb, dense, axis=1)
    x = x0
    for i in range(6):
        weight = init.random_normal((width, 1), stddev=0.001, name=f'dcn_cross{i}_weight')
        bias = init.zeros((width,), name=f'dcn_cross{i}_bias')
        cross = ht.mul_op(x0, ht.broadcastto_op(ht.matmul_op(x, weight), x0))
        x = x + cross + ht.broadcastto_op(bias, x0)

    deep = x0
    for i, input_width in enumerate((width, 1024)):
        weight = init.random_normal((input_width, 1024), stddev=0.001,
                                    name=f'dcn_deep{i}_weight')
        bias = init.zeros((1024,), name=f'dcn_deep{i}_bias')
        projected = ht.matmul_op(deep, weight)
        deep = ht.relu_op(projected + ht.broadcastto_op(bias, projected))
        deep = ht.dropout_op(deep, 0.5)
    combined = ht.concat_op(deep, x, axis=1)
    output_weight = init.random_normal((1024 + width, 1), stddev=0.001,
                                        name='dcn_output_weight')
    logits = ht.matmul_op(combined, output_weight)
    prediction = ht.sigmoid_op(logits)
    class_logits = ht.concat_op(logits * 0, logits, axis=1)
    class_labels = ht.concat_op((-1 * labels) + 1, labels, axis=1)
    loss = ht.reduce_mean_op(
        ht.softmaxcrossentropy_op(class_logits, class_labels, use_cudnn=False), [0])
    train = ht.optim.SGDOptimizer(learning_rate=learning_rate).minimize(loss)
    return loss, prediction, labels, train
