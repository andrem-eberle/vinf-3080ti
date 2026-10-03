#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace vinf {

template <typename T> __device__ inline float to_float(T value) {
    return static_cast<float>(value);
}

template <> __device__ inline float to_float<half>(half value) {
    return __half2float(value);
}

template <typename T> __device__ inline T from_float(float value) {
    return static_cast<T>(value);
}

template <> __device__ inline half from_float<half>(float value) {
    return __float2half(value);
}

__device__ inline float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ inline float block_reduce_sum(float value) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    float block_value = 0.0f;
    if (warp == 0) {
        block_value = lane < ((blockDim.x + 31) >> 5) ? warp_sums[lane] : 0.0f;
        block_value = warp_reduce_sum(block_value);
    }
    return block_value;
}

struct float4_pack {
    float x, y, z, w;
};

__device__ inline float4_pack load4(const float *ptr, int offset) {
    return {ptr[offset], ptr[offset + 1], ptr[offset + 2], ptr[offset + 3]};
}

__device__ inline void store4(float *ptr, int offset, const float4_pack &value) {
    ptr[offset] = value.x;
    ptr[offset + 1] = value.y;
    ptr[offset + 2] = value.z;
    ptr[offset + 3] = value.w;
}

} // namespace vinf
