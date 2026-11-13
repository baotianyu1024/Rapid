import hetu as ht
from hetu.launcher import launch
from hetu import init
from  hetu.gpu_ops.SharedTable import SharedTableOp
from hetu.communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t

from models.hetuctr_data import load_dataset
from models.hetuctr_models import WDL, DCN, DFM
from models.rapid_multihot_dcn import DCNV2, DCN as DCN_MULTI
from models.rapid_multihot_wdl_dfm import WDL_MULTI, DFM_MULTI
from models.rapid_multihot_dlrm import DLRM_MULTI

import os.path as osp
import numpy as np
import yaml
import time
import argparse
import json
import sys
from pathlib import Path
from sklearn import metrics

sys.setrecursionlimit(10000)

def comm_sync_data(comm, *args):
    array = ht.array(args, ht.cpu())
    comm.dlarrayNcclAllReduce(array, array, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum, comm.stream)
    comm.stream.sync()
    return array.asnumpy() / comm.nRanks.value

def worker(args):
    def train(iterations, auc_enabled=False):
        localiter = range(iterations)
        train_loss = []
        train_acc = []
        if auc_enabled:
            train_auc = []
        for it in localiter:
            if args.benchmark:
                loss_val, _ = executor.run('train', convert_to_numpy_ret_vals=True)
                train_loss.append(loss_val[0])
                continue
            loss_val, predict_y, y_val, _ = executor.run('train', convert_to_numpy_ret_vals=True)
            if y_val.shape[1] == 1: # for criteo case
                acc_val = np.equal(
                    y_val,
                    predict_y > 0.5).astype(float)
            else:
                acc_val = np.equal(
                    np.argmax(y_val, 1),
                    np.argmax(predict_y, 1)).astype(float)
            train_loss.append(loss_val[0])
            train_acc.append(acc_val)
            if auc_enabled:
                train_auc.append(metrics.roc_auc_score(y_val, predict_y))
        if args.benchmark:
            return np.mean(train_loss), 0, 0
        if auc_enabled:
            return np.mean(train_loss), np.mean(train_acc), np.mean(train_auc)
        else:
            return np.mean(train_loss), np.mean(train_acc), 0
    def validate(iterations):
        localiter = range(iterations)
        test_loss = []
        test_acc = []
        all_predictions = []
        all_labels = []
        for it in localiter:
            loss_val, test_y_predicted, y_test_val = executor.run('validate', convert_to_numpy_ret_vals=True)
            if y_test_val.shape[1] == 1: # for criteo case
                correct_prediction = np.equal(
                    y_test_val,
                    test_y_predicted > 0.5).astype(float)
            else:
                correct_prediction = np.equal(
                    np.argmax(y_test_val, 1),
                    np.argmax(test_y_predicted, 1)).astype(float)
            test_loss.append(loss_val[0])
            test_acc.append(correct_prediction)
            all_predictions.append(test_y_predicted.reshape(-1))
            all_labels.append(y_test_val.reshape(-1))
        local_labels = np.concatenate(all_labels).astype(np.float32)
        local_predictions = np.concatenate(all_predictions).astype(np.float32)
        packed = ht.array(np.concatenate([local_labels, local_predictions]), ht.cpu())
        gathered = ht.array(np.empty(len(local_labels) * 2 * nrank, dtype=np.float32), ht.cpu())
        comm.dlarrayAllGather(packed, gathered, ncclDataType_t.ncclFloat32, comm.stream)
        comm.stream.sync()
        by_rank = gathered.asnumpy().reshape(nrank, 2, len(local_labels))
        labels = by_rank[:, 0, :].reshape(-1)
        predictions = by_rank[:, 1, :].reshape(-1)
        return np.mean(test_loss), np.mean(test_acc), metrics.roc_auc_score(labels, predictions)

    def get_shard(data):
        part_size = data.shape[0] // nrank
        start = part_size * rank
        end = start + part_size if rank != nrank - 1 else data.shape[0]
        return data[start:end]

    def get_partitioned_shard(data):
        if data_arr is not None:
            return data[np.where(data_arr[:len(data)]==rank)]
        else:
            return get_shard(data)

    batch_size = args.batch_size

    nrank = comm.nRanks.value

    input_start = time.perf_counter()
    dense, sparse, labels = load_dataset(args.dataset, val=False)
    input_load_seconds = time.perf_counter() - input_start
    input_start = time.perf_counter()
    has_dense_feature = dense is not None
    dense_input = [[get_partitioned_shard(dense), batch_size, 'train']] if has_dense_feature else None
    sparse_input = [ht.Dataloader(get_partitioned_shard(sparse).astype(np.int64), batch_size, 'train', use_numpy=True)]
    if args.benchmark:
        print(f'INPUT_RANK rank={rank} rows={len(sparse_input[0].raw_data)}', flush=True)
    y_ = [[get_partitioned_shard(labels), batch_size, 'train']]

    if args.val:
        val_dense, val_sparse, val_labels = load_dataset(args.dataset, val=True)
        if rank == 0:
            print(f'VAL_INPUT rows={len(val_sparse)} local_rows={len(get_shard(val_sparse))} '
                  f'sparse_shape={val_sparse.shape} dense_shape={val_dense.shape}', flush=True)
        if has_dense_feature:
            dense_input.append([get_shard(val_dense), batch_size, 'validate'])
        sparse_input.append(ht.Dataloader(get_shard(val_sparse).astype(np.int64), batch_size, 'validate', use_numpy=True))
        y_.append([get_shard(val_labels), batch_size, 'validate'])

    dense_input = ht.dataloader_op(dense_input) if has_dense_feature else None
    sparse_input = ht.dataloader_op(sparse_input)
    y_ = ht.dataloader_op(y_)
    input_prepare_seconds = time.perf_counter() - input_start

    print("Data loaded.")
    models = {"wdl" : WDL, "dcn" : DCN, "dcn_multi": DCN_MULTI, "dfm" : DFM, "dcnv2": DCNV2,
              "wdl_multi": WDL_MULTI, "dfm_multi": DFM_MULTI, "dlrm_multi": DLRM_MULTI}
    loss, prediction, y_, train_op = models[args.model](args.dataset, dense_input, sparse_input, y_, args.embed_dim, rank, nrank, device_id,
        args.bound, root_arr, storage_arr)

    eval_nodes = {'train': [loss, train_op] if args.benchmark else [loss, prediction, y_, train_op]}
    if args.val:
        print('Validation enabled...')
        eval_nodes['validate'] = [loss, prediction, y_]
    executor = ht.Executor(eval_nodes, ctx=ht.gpu(device_id), comm_mode="AllReduce" if nrank > 1 else None, seed=123, log_path='./logs/')
    if args.val and rank == 0:
        print(f'VAL_BATCHES={executor.get_batch_num("validate")}', flush=True)

    if rank == 0:
        log_file = open(args.output, 'w')
    for ep in range(args.iter // args.log_every):
        if args.profile_io_prefix and ep == 3:
            io_start = {k: int(v) for k, v in (line.split(':', 1) for line in
                        Path('/proc/self/io').read_text().splitlines())}
        ep_st = time.time()
        train_loss, train_acc, train_auc = train(args.log_every)
        ep_en = time.time()
        if args.profile_io_prefix and ep == 10:
            io_end = {k: int(v) for k, v in (line.split(':', 1) for line in
                      Path('/proc/self/io').read_text().splitlines())}
            Path(f'{args.profile_io_prefix}.rank{rank}.json').write_text(json.dumps(
                dict(io_delta={k: io_end[k] - io_start[k] for k in io_start},
                     input_load_seconds=input_load_seconds,
                     input_prepare_seconds=input_prepare_seconds), indent=2) + '\n')
        if args.benchmark:
            elapsed = ht.array([ep_en - ep_st], ht.cpu())
            comm.dlarrayNcclAllReduce(elapsed, elapsed, ncclDataType_t.ncclFloat32,
                                     ncclRedOp_t.ncclMax, comm.stream)
            comm.stream.sync()
            train_time = elapsed.asnumpy()[0]
        else:
            train_time, train_loss, train_acc, train_auc = comm_sync_data(comm, ep_en - ep_st, train_loss, train_acc, train_auc)
        if rank==0:
            printstr = "TRAIN %d: loss %.4f acc %.4f time %.4f speed %d" % (ep * args.log_every, train_loss, train_acc, train_time, args.log_every*batch_size/train_time)
            print(printstr, flush=True)
            print(printstr, file=log_file, flush=True)
        if args.val and ep > 0 and ep % (args.eval_every // args.log_every) == 0:
            val_loss, val_acc, val_auc = validate(executor.get_batch_num('validate'))
            val_loss, val_acc, val_auc = comm_sync_data(comm, val_loss, val_acc, val_auc)
            if rank==0:
                printstr = "EVAL %d: val_loss %.4f val_acc %.4f val_auc %.4f" % (ep * args.log_every, val_loss, val_acc, val_auc)
                print(printstr, flush=True)
                print(printstr, file=log_file, flush=True)
                print('EVAL_METRICS=' + json.dumps(dict(step=(ep + 1) * args.log_every,
                    auc=float(val_auc), loss=float(val_loss), accuracy=float(val_acc))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="[criteo avazu company]")
    parser.add_argument("--model", type=str, required=True, help="[wdl dcn dfm]")
    parser.add_argument("--val", action="store_true", help="whether to use validation")
    parser.add_argument("--bound", type=int, default=10, help="cache bound")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--embed_dim", type=int, default=128, help="embedding dim")
    parser.add_argument("--iter", type=int, default=10000, help="nnumber of iteration")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--store_rate", type=float, default=0.01)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--output", type=str, default="hetuctr.log")
    parser.add_argument("--benchmark", action="store_true", help="measure steps without accuracy transfers")
    parser.add_argument("--profile-io-prefix", help="record process I/O for blocks 3 through 10")
    args = parser.parse_args()

    comm, device_id = ht.mpi_nccl_init()
    rank = comm.myRank.value
    if args.partition:
        args.partition = osp.normpath(osp.expanduser(args.partition))
        assert osp.exists(args.partition)
        partition = np.load(args.partition)
        data_arr = partition["data_partition"]
        root_arr = partition["embed_partition"]
        storage_arr = partition[str(rank)]
        storage_arr = storage_arr[:int(args.store_rate * len(storage_arr))]
        storage_arr = np.concatenate([np.where(root_arr==rank)[0], storage_arr])
    else:
        data_arr, root_arr, storage_arr = None, None, None

    worker(args)
    ht.mpi_nccl_finish(comm)
