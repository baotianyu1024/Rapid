#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>

int cudaMalloc(void **ptr, size_t size) {
    static int (*real_malloc)(void **, size_t);
    static int (*mem_info)(size_t *, size_t *);
    if (!real_malloc) {
        real_malloc = dlsym(RTLD_NEXT, "cudaMalloc");
        mem_info = dlsym(RTLD_NEXT, "cudaMemGetInfo");
    }
    int code = real_malloc(ptr, size);
    if (code) {
        size_t free_bytes = 0, total_bytes = 0;
        if (mem_info) mem_info(&free_bytes, &total_bytes);
        fprintf(stderr, "CUDA_MALLOC_FAILURE bytes=%zu free=%zu total=%zu code=%d\n",
                size, free_bytes, total_bytes, code);
    }
    return code;
}
