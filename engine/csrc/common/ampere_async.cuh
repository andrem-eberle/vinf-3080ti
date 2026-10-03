#pragma once

#include <cuda_runtime.h>

namespace vinf {

__device__ inline void ampere_cp_async_cg_16(void *shared_dst, const void *global_src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(shared_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
                 "l"(global_src)
                 : "memory");
#else
    char *dst = static_cast<char *>(shared_dst);
    const char *src = static_cast<const char *>(global_src);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        dst[i] = src[i];
    }
#endif
}

__device__ inline void ampere_cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ inline void ampere_cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}

} // namespace vinf

