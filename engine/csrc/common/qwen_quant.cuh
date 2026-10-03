// Shared qwen35 CUDA helpers: GGUF quant block decoding (ggml layouts, verified against
// vinf.gguf.dequant), reductions, and activations. Included by the per-op runtime
// (qwen_runtime.cu) and the fused megakernel (qwen_megakernel.cu); each translation unit
// gets its own copy (anonymous namespace), including the __constant__ IQ tables.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "qwen_iq_tables.h"

namespace {

enum TensorType : int {
    kF32 = 0,
    kF16 = 1,
    kQ8_0 = 8,
    kQ2_K = 10,
    kQ3_K = 11,
    kQ4_K = 12,
    kQ5_K = 13,
    kQ6_K = 14,
    kIQ2_XXS = 16,
    kIQ2_XS = 17,
    kIQ3_XXS = 18,
    kIQ2_S = 22,
    kIQ4_NL = 20,
    kIQ3_S = 21,
    kIQ4_XS = 23,
};

struct TypeTraits {
    int block_size;
    int type_size;
};

__host__ __device__ inline bool type_traits(int type, TypeTraits *out) {
    switch (type) {
        case kF32: *out = {1, 4}; return true;
        case kF16: *out = {1, 2}; return true;
        case kQ8_0: *out = {32, 34}; return true;
        case kQ2_K: *out = {256, 84}; return true;
        case kQ3_K: *out = {256, 110}; return true;
        case kQ4_K: *out = {256, 144}; return true;
        case kQ5_K: *out = {256, 176}; return true;
        case kQ6_K: *out = {256, 210}; return true;
        case kIQ4_NL: *out = {32, 18}; return true;
        case kIQ3_S: *out = {256, 110}; return true;
        case kIQ4_XS: *out = {256, 136}; return true;
        case kIQ2_XXS: *out = {256, 66}; return true;
        case kIQ2_XS: *out = {256, 74}; return true;
        case kIQ2_S: *out = {256, 82}; return true;
        case kIQ3_XXS: *out = {256, 98}; return true;        default: return false;
    }
}

__constant__ int8_t c_iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__constant__ uint8_t c_iq3s_grid[512 * 4];

// Device source of each type's codebook (IQ3_S's comes from Python at runtime init).
template <int TYPE>
__device__ __forceinline__ const uint8_t *grid_source() {
    if constexpr (TYPE == kIQ2_XXS) return c_iq2xxs_grid;
    else if constexpr (TYPE == kIQ2_XS) return c_iq2xs_grid;
    else if constexpr (TYPE == kIQ2_S) return c_iq2s_grid;
    else if constexpr (TYPE == kIQ3_XXS) return c_iq3xxs_grid;
    else return c_iq3s_grid;
}


__device__ __forceinline__ float f16_at(const uint8_t *p) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(p[0] | (p[1] << 8))));
}

__device__ __forceinline__ void k4_scale_min(const uint8_t *s, int g, int *sc, int *m) {
    if (g < 4) {
        *sc = s[g] & 63;
        *m = s[g + 4] & 63;
    } else {
        *sc = (s[g + 4] & 0x0F) | ((s[g - 4] >> 6) << 4);
        *m = (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
    }
}


__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// Block-wide reductions; result broadcast to all threads. blockDim.x must be a multiple of 32.
__device__ float block_sum(float v) {
    __shared__ float shared[32];
    __shared__ float result;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = lane < (blockDim.x + 31) / 32 ? shared[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) result = t;
    }
    __syncthreads();
    const float out = result;
    __syncthreads();
    return out;
}

__device__ float block_max(float v) {
    __shared__ float shared[32];
    __shared__ float result;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_max(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = lane < (blockDim.x + 31) / 32 ? shared[lane] : -INFINITY;
        t = warp_max(t);
        if (lane == 0) result = t;
    }
    __syncthreads();
    const float out = result;
    __syncthreads();
    return out;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }


// Dequantize 8 consecutive elements (chunk c, elements 8c..8c+7) of one quant block.
// Bytes of the codebook a type indexes (staged per block by the caller); 0 = none.
template <int TYPE>
constexpr int kGridBytes = TYPE == kIQ3_S ? 2048 : TYPE == kIQ2_XXS ? 2048 : TYPE == kIQ2_XS ? 4096
                         : TYPE == kIQ2_S ? 8192 : TYPE == kIQ3_XXS ? 1024 : 0;

__device__ __forceinline__ uint32_t u32_at(const uint8_t *p) {  // 2-byte aligned little-endian
    return static_cast<uint32_t>(p[0] | (p[1] << 8)) | (static_cast<uint32_t>(p[2] | (p[3] << 8)) << 16);
}

// `iq3s_grid` is the codebook of the block's type (IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS).
template <int TYPE>
__device__ __forceinline__ void deq8(const uint8_t *blk, int c, float v[8], const int8_t *iq4nl,
                                     const uint8_t *iq3s_grid) {
    if constexpr (TYPE == kQ8_0) {
        const float d = f16_at(blk);
        const int8_t *q = reinterpret_cast<const int8_t *>(blk + 2) + 8 * c;
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = d * q[i];
    } else if constexpr (TYPE == kIQ4_NL) {
        const float d = f16_at(blk);
        const uint8_t *qs = blk + 2 + 8 * (c % 2);
        const int shift = c >= 2 ? 4 : 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = d * iq4nl[(qs[i] >> shift) & 0x0F];
    } else if constexpr (TYPE == kQ4_K || TYPE == kQ5_K) {
        const float d = f16_at(blk), dmin = f16_at(blk + 2);
        const int g = c / 4, l0 = 8 * (c % 4);
        int sc, m;
        k4_scale_min(blk + 4, g, &sc, &m);
        const float dl = d * sc, ml = dmin * m;
        const int shift = 4 * (g % 2);
        if constexpr (TYPE == kQ4_K) {
            const uint8_t *q = blk + 16 + 32 * (g / 2) + l0;
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = dl * ((q[i] >> shift) & 0x0F) - ml;
        } else {
            const uint8_t *qh = blk + 16 + l0;
            const uint8_t *q = blk + 48 + 32 * (g / 2) + l0;
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = dl * (((q[i] >> shift) & 0x0F) + (((qh[i] >> g) & 1) << 4)) - ml;
        }
    } else if constexpr (TYPE == kQ6_K) {
        const int e0 = 8 * c;
        const int chunk = e0 / 128, quarter = (e0 % 128) / 32, l0 = e0 % 32;
        const uint8_t *ql = blk + chunk * 64 + (quarter % 2) * 32 + l0;
        const uint8_t *qh = blk + 128 + chunk * 32 + l0;
        const float dl = f16_at(blk + 208) * reinterpret_cast<const int8_t *>(blk + 192)[e0 / 16];
        const int lshift = quarter >= 2 ? 4 : 0, hshift = 2 * quarter;
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = dl * ((((ql[i] >> lshift) & 0x0F) | (((qh[i] >> hshift) & 3) << 4)) - 32);
    } else if constexpr (TYPE == kQ2_K) {
        const int e0 = 8 * c;
        const uint8_t sc = blk[e0 / 16];
        const float dl = f16_at(blk + 80) * (sc & 0x0F), ml = f16_at(blk + 82) * (sc >> 4);
        const uint8_t *qs = blk + 16 + 32 * (e0 / 128) + e0 % 32;
        const int shift = 2 * ((e0 % 128) / 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = dl * ((qs[i] >> shift) & 3) - ml;
    } else if constexpr (TYPE == kQ3_K) {
        const int e0 = 8 * c;
        const int k = e0 / 16;
        const uint8_t *s = blk + 96;
        const int lo = (s[k % 8] >> (4 * (k / 8))) & 0x0F;
        const int hi = (s[8 + k % 4] >> (2 * (k / 4))) & 0x03;
        const float dl = f16_at(blk + 108) * static_cast<float>((lo | (hi << 4)) - 32);
        const uint8_t *qs = blk + 32 + 32 * (e0 / 128) + e0 % 32;
        const uint8_t *hm = blk + e0 % 32;
        const int shift = 2 * ((e0 % 128) / 32), hbit = e0 / 32;
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = dl * (((qs[i] >> shift) & 3) - (((hm[i] >> hbit) & 1) ? 0 : 4));
    } else if constexpr (TYPE == kIQ4_XS) {
        const float d = f16_at(blk);
        const int g = c / 4, l0 = 8 * (c % 4);
        const int scales_h = blk[2] | (blk[3] << 8);
        const int ls = ((blk[4 + g / 2] >> (4 * (g % 2))) & 0x0F) | (((scales_h >> (2 * g)) & 3) << 4);
        const float dl = d * (ls - 32);
        const uint8_t *qs = blk + 8 + 16 * g + (l0 % 16);
        const int shift = l0 >= 16 ? 4 : 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = dl * iq4nl[(qs[i] >> shift) & 0x0F];
    } else if constexpr (TYPE == kIQ3_S) {
        const int e0 = 8 * c, g = c / 4;
        const float db = f16_at(blk) * (1 + 2 * ((blk[106 + g / 2] >> (4 * (g % 2))) & 0x0F));
        const uint8_t sign = blk[74 + e0 / 8];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int idx = e0 / 4 + half;
            const int entry = blk[2 + idx] | (((blk[66 + idx / 8] >> (idx % 8)) & 1) << 8);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int i = half * 4 + j;
                v[i] = db * iq3s_grid[entry * 4 + j] * (((sign >> i) & 1) ? -1.0f : 1.0f);
            }
        }
    } else if constexpr (TYPE == kIQ2_XXS || TYPE == kIQ2_XS || TYPE == kIQ2_S) {
        // ggml dequantize_row_iq2_{xxs,xs,s}: 8 weights = codebook row (8 magnitudes) x signs.
        const int ib32 = c / 4, l = c % 4;
        const float d = f16_at(blk);
        int entry, signs;
        float db;
        if constexpr (TYPE == kIQ2_XXS) {
            const uint32_t aux1 = u32_at(blk + 2 + 8 * ib32 + 4);
            entry = blk[2 + 8 * ib32 + l];
            signs = c_ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
            db = d * (0.5f + (aux1 >> 28)) * 0.25f;
        } else if constexpr (TYPE == kIQ2_XS) {
            const int q = blk[2 + 2 * (4 * ib32 + l)] | (blk[3 + 2 * (4 * ib32 + l)] << 8);
            entry = q & 511;
            signs = c_ksigns_iq2xs[q >> 9];
            db = d * (0.5f + ((blk[66 + ib32] >> (4 * (l / 2))) & 0x0F)) * 0.25f;
        } else {
            entry = blk[2 + 4 * ib32 + l] | ((blk[66 + ib32] << (8 - 2 * l)) & 0x300);
            signs = blk[34 + 4 * ib32 + l];
            db = d * (0.5f + ((blk[74 + ib32] >> (4 * (l / 2))) & 0x0F)) * 0.25f;
        }
        const uint8_t *g = iq3s_grid + entry * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = db * g[j] * (((signs >> j) & 1) ? -1.0f : 1.0f);
    } else if constexpr (TYPE == kIQ3_XXS) {
        const int ib32 = c / 4, l = c % 4;
        const uint32_t aux = u32_at(blk + 66 + 4 * ib32);
        const float db = f16_at(blk) * (0.5f + (aux >> 28)) * 0.5f;
        const int signs = c_ksigns_iq2xs[(aux >> (7 * l)) & 127];
        const uint8_t *g1 = iq3s_grid + blk[2 + 8 * ib32 + 2 * l] * 4;
        const uint8_t *g2 = iq3s_grid + blk[2 + 8 * ib32 + 2 * l + 1] * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] = db * g1[j] * (((signs >> j) & 1) ? -1.0f : 1.0f);
            v[j + 4] = db * g2[j] * (((signs >> (j + 4)) & 1) ? -1.0f : 1.0f);
        }
    } else if constexpr (TYPE == kF32) {
        const float *w = reinterpret_cast<const float *>(blk) + 8 * c;
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = w[i];
    } else if constexpr (TYPE == kF16) {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = f16_at(blk + 16 * c + 2 * i);
    }
}


}  // namespace
