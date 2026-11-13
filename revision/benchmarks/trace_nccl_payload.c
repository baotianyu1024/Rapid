#define _GNU_SOURCE
#include <dlfcn.h>
#include <nccl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, BROADCAST, SEND, RECV, OP_COUNT };
static const char *op_names[OP_COUNT] = {"all_reduce", "all_gather", "reduce_scatter", "broadcast", "send", "recv"};
static unsigned long long calls[OP_COUNT], bytes[OP_COUNT];
static unsigned long long remote_calls[OP_COUNT], remote_bytes[OP_COUNT];

void rapid_nccl_reset(void) {
  for (int i = 0; i < OP_COUNT; ++i) {
    __atomic_store_n(&calls[i], 0, __ATOMIC_RELAXED);
    __atomic_store_n(&bytes[i], 0, __ATOMIC_RELAXED);
    __atomic_store_n(&remote_calls[i], 0, __ATOMIC_RELAXED);
    __atomic_store_n(&remote_bytes[i], 0, __ATOMIC_RELAXED);
  }
}

static size_t dtype_bytes(ncclDataType_t dtype) {
  switch (dtype) {
    case ncclInt8: case ncclUint8: return 1;
    case ncclFloat16: case ncclBfloat16: return 2;
    case ncclInt32: case ncclUint32: case ncclFloat32: return 4;
    case ncclInt64: case ncclUint64: case ncclFloat64: return 8;
    default: return 0;
  }
}

static void record(int op, size_t count, ncclDataType_t dtype) {
  size_t nbytes = count * dtype_bytes(dtype);
  __atomic_fetch_add(&calls[op], 1, __ATOMIC_RELAXED);
  __atomic_fetch_add(&bytes[op], nbytes, __ATOMIC_RELAXED);
}

static void record_peer(int op, size_t count, ncclDataType_t dtype, int peer, ncclComm_t comm) {
  static ncclResult_t (*user_rank)(const ncclComm_t, int *);
  if (!user_rank) user_rank = dlsym(RTLD_NEXT, "ncclCommUserRank");
  record(op, count, dtype);
  int rank = -1;
  if (user_rank && user_rank(comm, &rank) == ncclSuccess && rank != peer) {
    __atomic_fetch_add(&remote_calls[op], 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&remote_bytes[op], count * dtype_bytes(dtype), __ATOMIC_RELAXED);
  }
}

static void report(void) {
  const char *rank = getenv("RANK");
  for (int i = 0; i < OP_COUNT; ++i)
    fprintf(stderr, "NCCL_PAYLOAD rank=%s op=%s calls=%llu logical_bytes=%llu\n",
            rank ? rank : "single-process", op_names[i], calls[i], bytes[i]);
  for (int i = SEND; i <= RECV; ++i)
    fprintf(stderr, "NCCL_REMOTE rank=%s op=%s calls=%llu logical_bytes=%llu\n",
            rank ? rank : "single-process", op_names[i], remote_calls[i], remote_bytes[i]);
}

__attribute__((constructor)) static void setup(void) { atexit(report); }

ncclResult_t ncclAllReduce(const void *src, void *dst, size_t count, ncclDataType_t dtype,
                           ncclRedOp_t redop, ncclComm_t comm, cudaStream_t stream) {
  static ncclResult_t (*next)(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
  if (!next) next = dlsym(RTLD_NEXT, "ncclAllReduce");
  record(ALL_REDUCE, count, dtype);
  return next(src, dst, count, dtype, redop, comm, stream);
}

ncclResult_t ncclAllGather(const void *src, void *dst, size_t count, ncclDataType_t dtype,
                           ncclComm_t comm, cudaStream_t stream) {
  static ncclResult_t (*next)(const void *, void *, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
  if (!next) next = dlsym(RTLD_NEXT, "ncclAllGather");
  record(ALL_GATHER, count, dtype);
  return next(src, dst, count, dtype, comm, stream);
}

ncclResult_t ncclReduceScatter(const void *src, void *dst, size_t count, ncclDataType_t dtype,
                                ncclRedOp_t redop, ncclComm_t comm, cudaStream_t stream) {
  static ncclResult_t (*next)(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
  if (!next) next = dlsym(RTLD_NEXT, "ncclReduceScatter");
  record(REDUCE_SCATTER, count, dtype);
  return next(src, dst, count, dtype, redop, comm, stream);
}

ncclResult_t ncclBroadcast(const void *src, void *dst, size_t count, ncclDataType_t dtype,
                           int root, ncclComm_t comm, cudaStream_t stream) {
  static ncclResult_t (*next)(const void *, void *, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
  if (!next) next = dlsym(RTLD_NEXT, "ncclBroadcast");
  record(BROADCAST, count, dtype);
  return next(src, dst, count, dtype, root, comm, stream);
}

ncclResult_t ncclSend(const void *src, size_t count, ncclDataType_t dtype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
  static ncclResult_t (*next)(const void *, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
  if (!next) next = dlsym(RTLD_NEXT, "ncclSend");
  record_peer(SEND, count, dtype, peer, comm);
  return next(src, count, dtype, peer, comm, stream);
}

ncclResult_t ncclRecv(void *dst, size_t count, ncclDataType_t dtype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
  static ncclResult_t (*next)(void *, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
  if (!next) next = dlsym(RTLD_NEXT, "ncclRecv");
  record_peer(RECV, count, dtype, peer, comm);
  return next(dst, count, dtype, peer, comm, stream);
}
