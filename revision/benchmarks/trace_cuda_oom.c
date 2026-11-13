#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <cuda_runtime_api.h>

cudaError_t cudaMalloc(void **ptr, size_t bytes) {
  static cudaError_t (*next_malloc)(void **, size_t);
  if (!next_malloc) next_malloc = dlsym(RTLD_NEXT, "cudaMalloc");
  cudaError_t result = next_malloc(ptr, bytes);
  if (result != cudaSuccess) {
    int device = -1;
    size_t free_bytes = 0, total_bytes = 0;
    cudaGetDevice(&device);
    cudaMemGetInfo(&free_bytes, &total_bytes);
    fprintf(stderr, "CUDA_ALLOC_FAILURE device=%d requested=%zu free=%zu total=%zu error=%d\n",
            device, bytes, free_bytes, total_bytes, result);
  }
  return result;
}
