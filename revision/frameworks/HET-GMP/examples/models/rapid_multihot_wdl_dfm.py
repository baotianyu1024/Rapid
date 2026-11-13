"""Multi-hot Wide&Deep and DeepFM using HET-GMP's shared embedding table."""
import os

import hetu as ht
from hetu import init
from hetu.gpu_ops.SharedTable import SharedTableOp


HOT = [int(value) for value in os.environ['RAPID_HET_HOTNESS'].split(',')]


def pooled_fields(sparse, embed_dim, rank, nrank, device_id, bound, root, storage, lr):
    width = embed_dim + 4
    vocab = int(os.environ['RAPID_HET_VOCAB'])
    if nrank == 1:
        table = init.random_normal((vocab, width), stddev=0.01, ctx=ht.gpu(device_id))
        features = ht.embedding_lookup_op(table, sparse, ctx=ht.gpu(device_id))
    else:
        features = SharedTableOp(sparse, rank, nrank, (vocab, width), lr,
                                 bounds=(bound, bound), root_arr=root, storage_arr=storage,
                                 ctx=ht.gpu(device_id))
    vectors = []
    linear = []
    start = 0
    for hot in HOT:
        values = ht.slice_op(features, (0, start, 0), (-1, hot, width))
        pooled = ht.array_reshape_op(values, (-1, width)) if hot == 1 else ht.reduce_sum_op(values, axes=1)
        vectors.append(ht.slice_op(pooled, (0, 0), (-1, embed_dim)))
        linear.append(ht.slice_op(pooled, (0, embed_dim), (-1, 1)))
        start += hot
    flat = vectors[0]
    sparse_linear = linear[0]
    for vector, first_order in zip(vectors[1:], linear[1:]):
        flat = ht.concat_op(flat, vector, axis=1)
        sparse_linear = sparse_linear + first_order
    return flat, sparse_linear


def dense_layer(x, input_width, output_width, name, relu=True):
    weight = init.random_normal((input_width, output_width), stddev=0.01,
                                name=name + '_weight')
    bias = init.zeros((output_width,), name=name + '_bias')
    result = ht.matmul_op(x, weight)
    result = result + ht.broadcastto_op(bias, result)
    return ht.relu_op(result) if relu else result


def build(name, dense, sparse, labels, embed_dim, rank, nrank, device_id,
          bound, root, storage, deepfm):
    dense_width = int(os.environ.get('RAPID_HET_DENSE_DIM', '13'))
    lr = float(os.environ.get('RAPID_HET_LR', '0.01'))
    flat, sparse_linear = pooled_fields(sparse, embed_dim, rank, nrank,
                                        device_id, bound, root, storage, lr)
    features = ht.concat_op(flat, dense, axis=1)
    width = len(HOT) * embed_dim + dense_width
    hidden = (400, 400, 400) if deepfm else (1024, 1024)
    x = features
    for index, output_width in enumerate(hidden):
        x = dense_layer(x, width, output_width, f'{name}_deep{index}')
        width = output_width
    deep_logit = dense_layer(x, width, 1, f'{name}_output', relu=False)
    dense_linear = dense_layer(dense, dense_width, 1, f'{name}_dense_linear', relu=False)
    logit = deep_logit + sparse_linear + dense_linear
    if deepfm:
        fields = ht.array_reshape_op(flat, (-1, len(HOT), embed_dim))
        summed = ht.reduce_sum_op(fields, axes=1)
        square_of_sum = ht.mul_op(summed, summed)
        sum_of_square = ht.reduce_sum_op(ht.mul_op(fields, fields), axes=1)
        interaction = (square_of_sum + -1 * sum_of_square) * 0.5
        fm_logit = ht.reduce_sum_op(interaction, axes=1, keepdims=True)
        logit = logit + fm_logit
    prediction = ht.sigmoid_op(logit)
    if deepfm:
        # The multi-hot FM logit can saturate sigmoid; use stable log-sum-exp BCE.
        logits = ht.concat_op(logit * 0, logit, axis=1)
        one_hot = ht.one_hot_op(ht.array_reshape_op(labels, (-1,)), 2)
        loss = ht.reduce_mean_op(
            ht.softmaxcrossentropy_op(logits, one_hot, use_cudnn=False), [0])
    else:
        loss = ht.reduce_mean_op(ht.binarycrossentropy_op(prediction, labels), [0])
    train = ht.optim.SGDOptimizer(learning_rate=float(os.environ.get('RAPID_HET_LR', '0.01'))).minimize(loss)
    return loss, prediction, labels, train


def WDL_MULTI(*args):
    return build(*args, deepfm=False)


def DFM_MULTI(*args):
    return build(*args, deepfm=True)
