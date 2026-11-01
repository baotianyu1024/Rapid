"""
HugeCTR DLRM training: original vs remapped data comparison.
Usage:
  python3.10 run_hugectr_dlrm_compare.py --mode original
  python3.10 run_hugectr_dlrm_compare.py --mode remapped
"""

import os
import sys
import argparse

if os.environ.get("HUGECTR_PYTHONPATH"):
    sys.path.insert(0, os.environ["HUGECTR_PYTHONPATH"])

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["original", "remapped", "original_truncated", "remapped_shuffled", "remapped_interleaved", "remapped_global_shuffled", "original_inflated", "remapped_v2", "original_full", "mapped_full"])
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--ev_size", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import hugectr

# ===================== Config per mode =====================

MULTI_HOT_SIZES = [
    3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1,
]

if args.mode == "original":
    TRAIN_DATA = "data/hugectr_raw_day7/day_7_multihot_shuffled_fixed.bin"
    NUM_SAMPLES = 20816207
    TABLE_SIZE_ARRAY = [
        40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
        40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14,
        40000000, 40000000, 40000000, 590152, 12973, 108, 36,
    ]
elif args.mode == "original_truncated":
    TRAIN_DATA = "data/hugectr_raw_day7/day_7_truncated.bin"
    NUM_SAMPLES = 13107200
    TABLE_SIZE_ARRAY = [
        40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
        40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14,
        40000000, 40000000, 40000000, 590152, 12973, 108, 36,
    ]
elif args.mode == "original_inflated":
    # Same data as original_truncated but with inflated TABLE_SIZE matching remapped
    TRAIN_DATA = "data/hugectr_raw_day7/day_7_truncated.bin"
    NUM_SAMPLES = 13107200
    TABLE_SIZE_ARRAY = [
        37703024, 191472, 76216, 47224, 159048, 24, 34112, 6744, 280,
        58395144, 3341800, 2545392, 48, 17400, 94472, 928, 24, 4768, 56,
        131215192, 237114800, 202705848, 3798800, 84632, 448, 184,
    ]
elif args.mode == "remapped":
    TRAIN_DATA = "data/dataset_output_8dev/merged.bin"
    NUM_SAMPLES = 13107200
    TABLE_SIZE_ARRAY = [
        37703024, 191472, 76216, 47224, 159048, 24, 34112, 6744, 280,
        58395144, 3341800, 2545392, 48, 17400, 94472, 928, 24, 4768, 56,
        131215192, 237114800, 202705848, 3798800, 84632, 448, 184,
    ]
elif args.mode == "remapped_v2":
    TRAIN_DATA = "data/dataset_output_8dev_v2/merged_global_shuffled.bin"
    NUM_SAMPLES = 13107200
    TABLE_SIZE_ARRAY = [
        5763456, 39552, 20960, 7912, 32856, 8, 9752, 1792, 48,
        8019896, 505456, 444952, 8, 4760, 24440, 184, 8, 728, 8,
        20097784, 35442288, 30440800, 556840, 23184, 120, 40,
    ]
elif args.mode == "remapped_global_shuffled":
    TRAIN_DATA = "data/dataset_output_8dev/merged_global_shuffled.bin"
    NUM_SAMPLES = 13107200
    TABLE_SIZE_ARRAY = [
        37703024, 191472, 76216, 47224, 159048, 24, 34112, 6744, 280,
        58395144, 3341800, 2545392, 48, 17400, 94472, 928, 24, 4768, 56,
        131215192, 237114800, 202705848, 3798800, 84632, 448, 184,
    ]
elif args.mode == "remapped_shuffled":
    TRAIN_DATA = "data/dataset_output_8dev/merged_shuffled.bin"
    NUM_SAMPLES = 13107200
    TABLE_SIZE_ARRAY = [
        37703024, 191472, 76216, 47224, 159048, 24, 34112, 6744, 280,
        58395144, 3341800, 2545392, 48, 17400, 94472, 928, 24, 4768, 56,
        131215192, 237114800, 202705848, 3798800, 84632, 448, 184,
    ]
elif args.mode == "original_full":
    TRAIN_DATA = "data/hugectr_train_8gpu/original.bin"
    NUM_SAMPLES = 20774912
    TABLE_SIZE_ARRAY = [
        40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
        40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14,
        40000000, 40000000, 40000000, 590152, 12973, 108, 36,
    ]
elif args.mode == "mapped_full":
    TRAIN_DATA = "data/hugectr_train_8gpu/mapped.bin"
    NUM_SAMPLES = 20774912
    TABLE_SIZE_ARRAY = [
        40000000, 40968, 20696, 7984, 32856, 16, 9392, 1576, 64,
        40000000, 3067956, 473672, 16, 4584, 23960, 184, 8, 976, 14,
        40000000, 40000000, 40000000, 605144, 22568, 108, 36,
    ]
else:  # remapped_interleaved
    TRAIN_DATA = "data/dataset_output_8dev/merged_interleaved.bin"
    NUM_SAMPLES = 13107200
    TABLE_SIZE_ARRAY = [
        37703024, 191472, 76216, 47224, 159048, 24, 34112, 6744, 280,
        58395144, 3341800, 2545392, 48, 17400, 94472, 928, 24, 4768, 56,
        131215192, 237114800, 202705848, 3798800, 84632, 448, 184,
    ]

IS_DENSE_FLOAT = args.mode in ("original_full", "mapped_full")

BATCH_SIZE_PER_GPU = 8192
NUM_GPUS = 8
BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS
NUM_TABLE = len(TABLE_SIZE_ARRAY)
NUM_DENSE = 13
EV_SIZE = args.ev_size

samples_per_epoch = NUM_SAMPLES // BATCH_SIZE * BATCH_SIZE
iters_per_epoch = NUM_SAMPLES // BATCH_SIZE
max_iter = iters_per_epoch * args.num_epochs

print(f"=== Mode: {args.mode} ===")
print(f"Data: {TRAIN_DATA}")
print(f"Samples: {NUM_SAMPLES}, Batch: {BATCH_SIZE}, Iters/epoch: {iters_per_epoch}")
print(f"Epochs: {args.num_epochs}, Total iters: {max_iter}")
print(f"EV_SIZE: {EV_SIZE}")
total_params = sum(TABLE_SIZE_ARRAY) * EV_SIZE
print(f"Total embedding params: {total_params:,} ({total_params * 4 / 1e9:.1f} GB)")
print(f"Per GPU: {total_params * 4 / NUM_GPUS / 1e9:.1f} GB")
print()

# ===================== HugeCTR Training =====================

solver = hugectr.CreateSolver(
    model_name=f"dlrm_{args.mode}",
    seed=args.seed,
    max_eval_batches=100,
    batchsize_eval=BATCH_SIZE,
    batchsize=BATCH_SIZE,
    vvgpu=[list(range(NUM_GPUS))],
    repeat_dataset=True,
    lr=0.01,
    warmup_steps=300,
    decay_start=1000,
    decay_steps=5000,
    decay_power=2.0,
    end_lr=0.0,
    use_mixed_precision=False,
    enable_tf32_compute=True,
    use_cuda_graph=False,
    gen_loss_summary=True,
    train_intra_iteration_overlap=True,
    train_inter_iteration_overlap=True,
    eval_intra_iteration_overlap=False,
    eval_inter_iteration_overlap=False,
    all_reduce_algo=hugectr.AllReduceAlgo.NCCL,
    num_iterations_statistics=20,
    use_embedding_collection=True,
)

optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.SGD,
    update_type=hugectr.Update_t.Local,
    atomic_update=True,
)

reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.RawAsync,
    source=[TRAIN_DATA],
    eval_source=TRAIN_DATA,
    check_type=hugectr.Check_t.Non,
    num_samples=NUM_SAMPLES,
    eval_num_samples=NUM_SAMPLES,
    slot_size_array=TABLE_SIZE_ARRAY,
    async_param=hugectr.AsyncParam(
        num_threads=1,
        num_batches_per_thread=3,
        shuffle=False,
        multi_hot_reader=True,
        is_dense_float=IS_DENSE_FLOAT,
    ),
)

model = hugectr.Model(solver, reader, optimizer)

model.add(
    hugectr.Input(
        label_dim=1,
        label_name="label",
        dense_dim=NUM_DENSE,
        dense_name="dense",
        data_reader_sparse_param_array=[
            hugectr.DataReaderSparseParam("data{}".format(i), MULTI_HOT_SIZES[i], True, 1)
            for i in range(NUM_TABLE)
        ],
    )
)

embedding_table_list = []
for i in range(NUM_TABLE):
    embedding_table_list.append(
        hugectr.EmbeddingTableConfig(
            name=str(i), max_vocabulary_size=TABLE_SIZE_ARRAY[i], ev_size=EV_SIZE
        )
    )

ebc_config = hugectr.EmbeddingCollectionConfig(
    use_exclusive_keys=True,
    comm_strategy=hugectr.CommunicationStrategy.Uniform,
)
ebc_config.embedding_lookup(
    table_config=[embedding_table_list[i] for i in range(NUM_TABLE)],
    bottom_name=["data{}".format(i) for i in range(NUM_TABLE)],
    top_name="sparse_embedding",
    combiner=["sum" for _ in range(NUM_TABLE)],
)

shard_matrix = [[str(i) for i in range(NUM_TABLE)] for _ in range(NUM_GPUS)]
shard_strategy = [("mp", [str(i) for i in range(NUM_TABLE)])]
ebc_config.shard(shard_matrix=shard_matrix, shard_strategy=shard_strategy)

model.add(ebc_config)

compute_config = hugectr.DenseLayerComputeConfig(
    async_wgrad=True,
    fuse_wb=False,
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MLP,
        bottom_names=["dense"],
        top_names=["mlp1"],
        num_outputs=[512, 256, 128],
        act_type=hugectr.Activation_t.Relu,
        compute_config=compute_config,
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Concat,
        bottom_names=["sparse_embedding", "mlp1"],
        top_names=["concat1"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MultiCross,
        bottom_names=["concat1"],
        top_names=["interaction1"],
        projection_dim=512,
        num_layers=3,
        compute_config=compute_config,
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MLP,
        bottom_names=["interaction1"],
        top_names=["mlp2"],
        num_outputs=[1024, 1024, 512, 256, 1],
        activations=[
            hugectr.Activation_t.Relu,
            hugectr.Activation_t.Relu,
            hugectr.Activation_t.Relu,
            hugectr.Activation_t.Relu,
            hugectr.Activation_t.Non,
        ],
        compute_config=compute_config,
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["mlp2", "label"],
        top_names=["loss"],
    )
)

model.compile()
model.summary()

if args.mode in ("original_full", "mapped_full"):
    eval_interval = 100
else:
    eval_interval = max(iters_per_epoch // 3, 50)
print(f"\nTraining: max_iter={max_iter}, eval_interval={eval_interval}, display=50")

model.fit(
    max_iter=max_iter,
    display=50,
    eval_interval=eval_interval,
    snapshot=10000000,
    snapshot_prefix=f"dlrm_{args.mode}",
)
