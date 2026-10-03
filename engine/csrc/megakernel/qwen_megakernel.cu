// qwen35 fused decode megakernel (Phase 31b).
//
// One cooperative launch runs one token. Grid = one block per SM (all co-resident);
// block b executes instruction queue b in order. Rows follow qwen_mk_abi.h:
// [op, wait0 counter, wait0 target, wait1 counter, wait1 target, signal, params...].
//
// Dependencies: counters are zeroed before each launch. Before an instruction, thread 0
// spins until each wait counter reaches its target; after it, thread 0 increments the
// signal counter (after __syncthreads + __threadfence). Queues preserve one global
// topological order, so the earliest unfinished instruction can always run (no deadlock).
//
// Coherence: this module is compiled with -Xptxas -dlcm=cg, so global loads bypass L1;
// data written by other SMs during the launch (activations, streamed weight slots) is
// read from L2.
//
// Ops:
//   QMV       y[r] (+)= W[r] . x for r in [row_start, row_end); x staged in shared memory,
//             optionally rmsnorm'd or silu(x)*x2; optional per-instruction argmax partial.
//             Weights come from the tensor table (resident) or a streaming slot.
//   LOAD      copies a byte range of a host-mapped streamed tensor into a slot.
//   ATTNHEAD  one query head: q split/norm/RoPE, k/v norm/RoPE (+cache append by the
//             group's first head), causal softmax attention, sigmoid output gate.
//   SSMGROUP  one key head and its tiled value heads: causal conv, L2-normed q/k,
//             gated delta rule on device state, gated RMSNorm.
//   ARGMAX    reduces QMV argmax partials to the output token.
//
// Streaming: by default the copy engine fills ring slots concurrently with the kernel via a
// host-enqueued copy plan on a non-blocking stream: cuStreamWaitValue32(consumed counter of
// the slot's previous item) -> cudaMemcpyAsync -> cuStreamWriteValue32(ready counter = 1).
// The kernel waits on the ready counter like any other dependency. DMA reaches PCIe rate
// (~21 GB/s here) where SM loads of host-mapped memory (LOAD op, fallback) top out ~13 GB/s.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "config_3080ti.cuh"
#include "qwen_mk_abi.h"
#include "qwen_quant.cuh"

namespace {

#ifndef VINF_MK_WARPS
#define VINF_MK_WARPS 16
#endif
// Warps per SM block (one block per SM). Matvec is memory-latency bound, so more warps in
// flight raise bandwidth; bounded by registers (65536 / threads per SM).
constexpr int kThreads = VINF_MK_WARPS * 32;
constexpr int kWarps = kThreads / 32;

struct MkTensor {
    const uint8_t *ptr;
    int type;
    int rows;
    int cols;
    int row_bytes;
};

struct MkParams {
    int position;
    int max_ctx;
    int heads;
    int kv_heads;
    int hd;
    int rot;
    double freq_base;
    float eps;
    int key_heads;
    int value_heads;
    int kd;
    int vd;
    int conv_k;
    int out_token;
    int error;
    int error_block;
    int error_instr;
    long long timeout_cycles;
};

struct MkArgs {
    const int *instr;
    int queue_len;
    const MkTensor *tensors;
    float *const *buffers;
    uint8_t *const *slots;
    int *counters;
    float *partial_val;
    int *partial_idx;
    MkParams *params;
    unsigned long long *timing;  // optional [blocks][queue_len][3]: wait start, exec start, exec end (ns)
};

__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ bool aborted(MkParams *p) { return *reinterpret_cast<volatile int *>(&p->error) != 0; }

// Thread 0 only. Returns false on timeout or global abort.
__device__ bool wait_counter(const MkArgs &a, int counter, int target, int instr) {
    if (counter < 0) return true;
    volatile int *c = a.counters + counter;
    const long long start = clock64();
    while (*c < target) {
        if (aborted(a.params)) return false;
        if (clock64() - start > a.params->timeout_cycles) {
            if (atomicCAS(&a.params->error, 0, 1) == 0) {
                a.params->error_block = blockIdx.x;
                a.params->error_instr = instr;
            }
            return false;
        }
        __nanosleep(64);
    }
    __threadfence();
    return true;
}

__device__ __forceinline__ void signal_counter(const MkArgs &a, int counter) {
    if (counter < 0) return;
    __threadfence();
    atomicAdd(a.counters + counter, 1);
}

// ---- QMV ---------------------------------------------------------------------------------

#ifndef VINF_MK_WARP_BUF
#define VINF_MK_WARP_BUF 1024
#endif
constexpr int kWarpBufBytes = VINF_MK_WARP_BUF;  // per-warp shared staging of weight bytes
constexpr int kVecPerLane = kWarpBufBytes / 16 / 32;  // 16-byte vectors each lane moves per refill

__device__ __forceinline__ float dot8(const float v[8], const float *x) {
    const float4 a = *reinterpret_cast<const float4 *>(x);
    const float4 b = *reinterpret_cast<const float4 *>(x + 4);
    return v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w + v[4] * b.x + v[5] * b.y + v[6] * b.z + v[7] * b.w;
}

// Each warp owns rows r0+warp, r0+warp+kWarps, ... and walks (row, block-group) work items.
// A work item is a run of whole quant blocks (<= kWarpBufBytes). Lanes fetch the item's bytes
// as aligned 16-byte vectors (coalesced; L2 via -dlcm=cg) into registers one item ahead, store
// them to the warp's shared buffer, and decode 8-element chunks from shared memory while the
// next item's loads are in flight. Reads may extend up to 15 bytes past a row/tensor end;
// device weight allocations are padded for that.
template <int TYPE>
__device__ void qmv_rows(const uint8_t *w, int row_bytes, int cols, int r0, int r1, const float *xs, float *y,
                         bool add, bool track, float *best_v, int *best_i, const int8_t *iq4nl,
                         const uint8_t *iq3s, uint8_t *warp_buf) {
    TypeTraits t;
    type_traits(TYPE, &t);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int bs = t.block_size >= 8 ? t.block_size : 8;  // pseudo-blocks of 8 for unquantized types
    const int bbytes = t.block_size >= 8 ? t.type_size : 8 * t.type_size;
    const int cpb = bs / 8;
    const int nblocks = cols / bs;
    const int group = (kWarpBufBytes - 16) / bbytes;
    const int groups_per_row = (nblocks + group - 1) / group;
    const int my_rows = r1 > r0 + warp ? (r1 - r0 - warp + kWarps - 1) / kWarps : 0;
    const int items = my_rows * groups_per_row;
    uint8_t *buf = warp_buf + warp * kWarpBufBytes;

    auto item_src = [&](int item, int *delta, int *nvec, int *nb) -> const uint4 * {
        const int r = r0 + warp + (item / groups_per_row) * kWarps;
        const int b0 = (item % groups_per_row) * group;
        *nb = min(group, nblocks - b0);
        const uint8_t *src = w + static_cast<size_t>(r) * row_bytes + static_cast<size_t>(b0) * bbytes;
        const uintptr_t aligned = reinterpret_cast<uintptr_t>(src) & ~static_cast<uintptr_t>(15);
        *delta = static_cast<int>(reinterpret_cast<uintptr_t>(src) - aligned);
        *nvec = (*delta + *nb * bbytes + 15) / 16;
        return reinterpret_cast<const uint4 *>(aligned);
    };

    uint4 pre[kVecPerLane];
    int delta = 0, nvec = 0, nb = 0;
    if (items > 0) {
        const uint4 *s4 = item_src(0, &delta, &nvec, &nb);
#pragma unroll
        for (int u = 0; u < kVecPerLane; ++u) {
            const int i = lane + 32 * u;
            if (i < nvec) pre[u] = s4[i];
        }
    }
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int item = 0; item < items; ++item) {
        uint4 *d4 = reinterpret_cast<uint4 *>(buf);
#pragma unroll
        for (int u = 0; u < kVecPerLane; ++u) {
            const int i = lane + 32 * u;
            if (i < nvec) d4[i] = pre[u];
        }
        __syncwarp();
        const int cur_delta = delta, cur_nb = nb;
        const int b0 = (item % groups_per_row) * group;
        if (item + 1 < items) {  // prefetch the next item while this one decodes
            const uint4 *s4 = item_src(item + 1, &delta, &nvec, &nb);
#pragma unroll
            for (int u = 0; u < kVecPerLane; ++u) {
                const int i = lane + 32 * u;
                if (i < nvec) pre[u] = s4[i];
            }
        }
        const uint8_t *blk0 = buf + cur_delta;
        const int nchunks = cur_nb * cpb;
        const float *xg = xs + static_cast<size_t>(b0) * bs;
        int c = lane;
        for (; c + 32 < nchunks; c += 64) {
            float v0[8], v1[8];
            const int c1 = c + 32;
            deq8<TYPE>(blk0 + (c / cpb) * bbytes, c % cpb, v0, iq4nl, iq3s);
            deq8<TYPE>(blk0 + (c1 / cpb) * bbytes, c1 % cpb, v1, iq4nl, iq3s);
            acc0 += dot8(v0, xg + 8 * c);
            acc1 += dot8(v1, xg + 8 * c1);
        }
        if (c < nchunks) {
            float v[8];
            deq8<TYPE>(blk0 + (c / cpb) * bbytes, c % cpb, v, iq4nl, iq3s);
            acc0 += dot8(v, xg + 8 * c);
        }
        __syncwarp();
        if (item % groups_per_row == groups_per_row - 1) {
            const int r = r0 + warp + (item / groups_per_row) * kWarps;
            const float acc = warp_sum(acc0 + acc1);
            acc0 = acc1 = 0.0f;
            if (lane == 0) {
                const float out = add ? y[r] + acc : acc;
                y[r] = out;
                if (track && (out > *best_v || (out == *best_v && r < *best_i))) {
                    *best_v = out;
                    *best_i = r;
                }
            }
        }
    }
}

// ---- int8 activation path (dp4a) ------------------------------------------------------------
// x is quantized per 32 elements: xq (int8), xd (scale), xsum (= xd * sum(xq), for min terms).
// A lane computes one 32-element weight group against one x group with __dp4a.

template <int TYPE>
constexpr bool kInt8Path = TYPE == kQ8_0 || TYPE == kQ3_K || TYPE == kQ4_K || TYPE == kQ5_K || TYPE == kQ6_K ||
                           TYPE == kIQ4_NL || TYPE == kIQ4_XS || TYPE == kIQ3_S;

__host__ __device__ inline bool int8_path_type(int type) {
    return type == kQ8_0 || type == kQ3_K || type == kQ4_K || type == kQ5_K || type == kQ6_K || type == kIQ4_NL ||
           type == kIQ4_XS || type == kIQ3_S;
}

// 32-bit little-endian load from a 2-byte aligned address (all block fields are even-aligned).
__device__ __forceinline__ int ld32(const uint8_t *p) {
    const uint16_t *h = reinterpret_cast<const uint16_t *>(p);
    return static_cast<int>(static_cast<uint32_t>(h[0]) | (static_cast<uint32_t>(h[1]) << 16));
}

// Map four 4-bit indices (one per byte, values 0..15) through the IQ4 table to packed int8.
__device__ __forceinline__ int iq4_lookup4(int nib4, const int8_t *tab) {
    const uint32_t u = static_cast<uint32_t>(nib4);
    const uint32_t b0 = static_cast<uint8_t>(tab[u & 0xF]);
    const uint32_t b1 = static_cast<uint8_t>(tab[(u >> 8) & 0xF]);
    const uint32_t b2 = static_cast<uint8_t>(tab[(u >> 16) & 0xF]);
    const uint32_t b3 = static_cast<uint8_t>(tab[(u >> 24) & 0xF]);
    return static_cast<int>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
}

template <int TYPE>
__device__ __forceinline__ float dot32_q(const uint8_t *blk, int g, const int *xq_ptr, float xd, float xsum,
                                         const int8_t *iq4nl, const uint8_t *iq3s) {
    int xqi[8];
    {
        const int4 a = reinterpret_cast<const int4 *>(xq_ptr)[0];
        const int4 b = reinterpret_cast<const int4 *>(xq_ptr)[1];
        xqi[0] = a.x; xqi[1] = a.y; xqi[2] = a.z; xqi[3] = a.w;
        xqi[4] = b.x; xqi[5] = b.y; xqi[6] = b.z; xqi[7] = b.w;
    }
    if constexpr (TYPE == kQ8_0) {
        int s = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) s = __dp4a(ld32(blk + 2 + 4 * i), xqi[i], s);
        return f16_at(blk) * xd * s;
    } else if constexpr (TYPE == kIQ4_NL) {
        int s = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int w = ld32(blk + 2 + 4 * i);
            s = __dp4a(iq4_lookup4(w & 0x0F0F0F0F, iq4nl), xqi[i], s);
            s = __dp4a(iq4_lookup4((w >> 4) & 0x0F0F0F0F, iq4nl), xqi[i + 4], s);
        }
        return f16_at(blk) * xd * s;
    } else if constexpr (TYPE == kIQ4_XS) {
        // Blocks are 136 bytes, so block starts (and qs at +8+16g) are 8-byte aligned.
        const int scales_h = blk[2] | (blk[3] << 8);
        const int ls = ((blk[4 + g / 2] >> (4 * (g % 2))) & 0x0F) | (((scales_h >> (2 * g)) & 3) << 4);
        const int2 *qs2 = reinterpret_cast<const int2 *>(blk + 8 + 16 * g);
        const int2 qa = qs2[0], qb = qs2[1];
        const int qw[4] = {qa.x, qa.y, qb.x, qb.y};
        int s = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int w = qw[i];
            s = __dp4a(iq4_lookup4(w & 0x0F0F0F0F, iq4nl), xqi[i], s);
            s = __dp4a(iq4_lookup4((w >> 4) & 0x0F0F0F0F, iq4nl), xqi[i + 4], s);
        }
        return f16_at(blk) * (ls - 32) * xd * s;
    } else if constexpr (TYPE == kQ4_K || TYPE == kQ5_K) {
        // Blocks are 144 / 176 bytes: block starts and the qs/qh fields are 16-byte aligned.
        int sc, m;
        k4_scale_min(blk + 4, g, &sc, &m);
        const int shift = 4 * (g % 2);
        const int4 *qs4 = reinterpret_cast<const int4 *>(blk + (TYPE == kQ4_K ? 16 : 48) + 32 * (g / 2));
        const int4 q0 = qs4[0], q1 = qs4[1];
        const int qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        int hw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if constexpr (TYPE == kQ5_K) {
            const int4 *qh4 = reinterpret_cast<const int4 *>(blk + 16);
            const int4 h0 = qh4[0], h1 = qh4[1];
            hw[0] = h0.x; hw[1] = h0.y; hw[2] = h0.z; hw[3] = h0.w;
            hw[4] = h1.x; hw[5] = h1.y; hw[6] = h1.z; hw[7] = h1.w;
        }
        int s = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int v = (qw[i] >> shift) & 0x0F0F0F0F;
            if constexpr (TYPE == kQ5_K) v |= ((hw[i] >> g) & 0x01010101) << 4;
            s = __dp4a(v, xqi[i], s);
        }
        return f16_at(blk) * sc * xd * s - f16_at(blk + 2) * m * xsum;
    } else if constexpr (TYPE == kQ6_K) {
        const int chunk = g / 4, quarter = g % 4;
        const uint8_t *ql = blk + chunk * 64 + (quarter % 2) * 32;
        const uint8_t *qh = blk + 128 + chunk * 32;
        const int lshift = quarter >= 2 ? 4 : 0, hshift = 2 * quarter;
        const int8_t *scales = reinterpret_cast<const int8_t *>(blk + 192);
        int s_lo = 0, s_hi = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int v = ((ld32(ql + 4 * i) >> lshift) & 0x0F0F0F0F) | (((ld32(qh + 4 * i) >> hshift) & 0x03030303) << 4);
            const int q = __vsubss4(v, 0x20202020);
            if (i < 4) s_lo = __dp4a(q, xqi[i], s_lo); else s_hi = __dp4a(q, xqi[i], s_hi);
        }
        return f16_at(blk + 208) * xd * (scales[2 * g] * s_lo + scales[2 * g + 1] * s_hi);
    } else if constexpr (TYPE == kQ3_K) {
        const uint8_t *qs = blk + 32 + 32 * (g / 4);
        const int shift = 2 * (g % 4);
        int s_lo = 0, s_hi = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int v = ((ld32(qs + 4 * i) >> shift) & 0x03030303) | (((ld32(blk + 4 * i) >> g) & 0x01010101) << 2);
            const int q = __vsubss4(v, 0x04040404);
            if (i < 4) s_lo = __dp4a(q, xqi[i], s_lo); else s_hi = __dp4a(q, xqi[i], s_hi);
        }
        const uint8_t *sb = blk + 96;
        int sc[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k = 2 * g + h;
            const int lo = (sb[k % 8] >> (4 * (k / 8))) & 0x0F;
            const int hi = (sb[8 + k % 4] >> (2 * (k / 4))) & 0x03;
            sc[h] = (lo | (hi << 4)) - 32;
        }
        return f16_at(blk + 108) * xd * (sc[0] * s_lo + sc[1] * s_hi);
    } else if constexpr (TYPE == kIQ3_S) {
        // Grid entries are 4 packed uint8 values; sign bits flip bytes via (v ^ m) - m.
        const float db = f16_at(blk) * (1 + 2 * ((blk[106 + g / 2] >> (4 * (g % 2))) & 0x0F));
        const uint8_t *qs = blk + 2 + 8 * g;
        const int qh = blk[66 + g];
        const uint8_t *signs = blk + 74 + 4 * g;
        int s = 0;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int entry = qs[l] | (((qh >> l) & 1) << 8);
            const int v = *reinterpret_cast<const int *>(iq3s + entry * 4);
            const uint32_t bits = (signs[l / 2] >> (4 * (l % 2))) & 0xF;
            const int m = static_cast<int>(((bits * 0x00204081u) & 0x01010101u) * 0xFFu);
            s = __dp4a(static_cast<int>(__vsub4(static_cast<unsigned>(v ^ m), static_cast<unsigned>(m))), xqi[l], s);
        }
        return db * xd * s;
    }
    return 0.0f;
}

// Same work-item / prefetch structure as qmv_rows, but lanes take whole 32-element groups.
template <int TYPE>
__device__ void qmv_rows_q(const uint8_t *w, int row_bytes, int cols, int r0, int r1, const int8_t *xq,
                           const float *xd, const float *xsum, float *y, bool add, bool track, float *best_v,
                           int *best_i, const int8_t *iq4nl, const uint8_t *iq3s, uint8_t *warp_buf) {
    TypeTraits t;
    type_traits(TYPE, &t);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int bs = t.block_size, bbytes = t.type_size;
    const int gpb = bs / 32;
    const int nblocks = cols / bs;
    const int group = (kWarpBufBytes - 16) / bbytes;
    const int groups_per_row = (nblocks + group - 1) / group;
    const int my_rows = r1 > r0 + warp ? (r1 - r0 - warp + kWarps - 1) / kWarps : 0;
    const int items = my_rows * groups_per_row;
    uint8_t *buf = warp_buf + warp * kWarpBufBytes;

    auto item_src = [&](int item, int *delta, int *nvec, int *nb) -> const uint4 * {
        const int r = r0 + warp + (item / groups_per_row) * kWarps;
        const int b0 = (item % groups_per_row) * group;
        *nb = min(group, nblocks - b0);
        const uint8_t *src = w + static_cast<size_t>(r) * row_bytes + static_cast<size_t>(b0) * bbytes;
        const uintptr_t aligned = reinterpret_cast<uintptr_t>(src) & ~static_cast<uintptr_t>(15);
        *delta = static_cast<int>(reinterpret_cast<uintptr_t>(src) - aligned);
        *nvec = (*delta + *nb * bbytes + 15) / 16;
        return reinterpret_cast<const uint4 *>(aligned);
    };

    uint4 pre[kVecPerLane];
    int delta = 0, nvec = 0, nb = 0;
    if (items > 0) {
        const uint4 *s4 = item_src(0, &delta, &nvec, &nb);
#pragma unroll
        for (int u = 0; u < kVecPerLane; ++u) {
            const int i = lane + 32 * u;
            if (i < nvec) pre[u] = s4[i];
        }
    }
    float acc = 0.0f;
    for (int item = 0; item < items; ++item) {
        uint4 *d4 = reinterpret_cast<uint4 *>(buf);
#pragma unroll
        for (int u = 0; u < kVecPerLane; ++u) {
            const int i = lane + 32 * u;
            if (i < nvec) d4[i] = pre[u];
        }
        __syncwarp();
        const int cur_delta = delta, cur_nb = nb;
        const int b0 = (item % groups_per_row) * group;
        if (item + 1 < items) {
            const uint4 *s4 = item_src(item + 1, &delta, &nvec, &nb);
#pragma unroll
            for (int u = 0; u < kVecPerLane; ++u) {
                const int i = lane + 32 * u;
                if (i < nvec) pre[u] = s4[i];
            }
        }
        const uint8_t *blk0 = buf + cur_delta;
        const int ngroups = cur_nb * gpb;
        for (int gg = lane; gg < ngroups; gg += 32) {
            const int xg = b0 * gpb + gg;
            acc += dot32_q<TYPE>(blk0 + (gg / gpb) * bbytes, gg % gpb, reinterpret_cast<const int *>(xq + 32 * xg),
                                 xd[xg], xsum[xg], iq4nl, iq3s);
        }
        __syncwarp();
        if (item % groups_per_row == groups_per_row - 1) {
            const int r = r0 + warp + (item / groups_per_row) * kWarps;
            const float total = warp_sum(acc);
            acc = 0.0f;
            if (lane == 0) {
                const float out = add ? y[r] + total : total;
                y[r] = out;
                if (track && (out > *best_v || (out == *best_v && r < *best_i))) {
                    *best_v = out;
                    *best_i = r;
                }
            }
        }
    }
}

__device__ void op_qmv(const MkArgs &a, const int *row, float *smem, const int8_t *iq4nl, const uint8_t *iq3s) {
    const MkTensor t = a.tensors[row[VINF_MK_QMV_TENSOR]];
    const int slot = row[VINF_MK_QMV_SLOT];
    const uint8_t *w = slot >= 0 ? a.slots[slot] : t.ptr;
    const float *x = a.buffers[row[VINF_MK_QMV_X]];
    float *y = a.buffers[row[VINF_MK_QMV_Y]];
    const int flags = row[VINF_MK_QMV_FLAGS];
    const int cols = t.cols;
    const bool silu_mul = (flags & VINF_MK_FLAG_SILU_MUL) != 0;
    const float *x2 = silu_mul ? a.buffers[row[VINF_MK_QMV_X2]] : nullptr;
    const float *nw = (flags & VINF_MK_FLAG_NORM) ? reinterpret_cast<const float *>(a.tensors[row[VINF_MK_QMV_NORM]].ptr)
                                                  : nullptr;
    // Dynamic smem: [warp byte buffers | staged x]. Staged x is float (float path) or
    // int8 + per-32 scale and sum (int8 path).
    uint8_t *warp_buf = reinterpret_cast<uint8_t *>(smem);
    float *stage = smem + kWarps * kWarpBufBytes / sizeof(float);
    // Staging is latency-bound (x lives in L2), so every loop below issues its loads in batches
    // before using them.
    float norm_scale = 1.0f;
    if (nw != nullptr) {
        float local = 0.0f;
        constexpr int kU = 4;
        for (int base = threadIdx.x; base < cols; base += kU * blockDim.x) {
            float v[kU];
#pragma unroll
            for (int u = 0; u < kU; ++u) {
                const int i = base + u * blockDim.x;
                v[u] = i < cols ? (silu_mul ? silu_f(x[i]) * x2[i] : x[i]) : 0.0f;
            }
#pragma unroll
            for (int u = 0; u < kU; ++u) local += v[u] * v[u];
        }
        norm_scale = rsqrtf(block_sum(local) / cols + a.params->eps);
    }
    const bool int8_path = int8_path_type(t.type);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int8_t *xq = reinterpret_cast<int8_t *>(stage);
    float *xd = stage + cols / 4;  // after cols int8 values
    float *xsum = xd + cols / 32;
    if (int8_path) {
        constexpr int kG = 8;  // groups per warp per batch
        const int ngroups = cols / 32;
        for (int base = warp; base < ngroups; base += kG * kWarps) {
            float v[kG];
#pragma unroll
            for (int u = 0; u < kG; ++u) {
                const int gi = base + u * kWarps;
                const int i = gi * 32 + lane;
                v[u] = gi < ngroups ? (silu_mul ? silu_f(x[i]) * x2[i] : x[i]) : 0.0f;
                if (nw != nullptr && gi < ngroups) v[u] *= norm_scale * nw[i];
            }
#pragma unroll
            for (int u = 0; u < kG; ++u) {
                const int gi = base + u * kWarps;
                if (gi >= ngroups) break;
                const float amax = warp_max(fabsf(v[u]));
                const float d = amax / 127.0f;
                const int q = d > 0.0f ? __float2int_rn(v[u] / d) : 0;
                xq[gi * 32 + lane] = static_cast<int8_t>(q);
                const int qsum = static_cast<int>(warp_sum(static_cast<float>(q)));
                if (lane == 0) {
                    xd[gi] = d;
                    xsum[gi] = d * qsum;
                }
            }
        }
    } else {
        constexpr int kU = 4;
        for (int base = threadIdx.x; base < cols; base += kU * blockDim.x) {
            float v[kU];
#pragma unroll
            for (int u = 0; u < kU; ++u) {
                const int i = base + u * blockDim.x;
                v[u] = i < cols ? (silu_mul ? silu_f(x[i]) * x2[i] : x[i]) : 0.0f;
                if (nw != nullptr && i < cols) v[u] *= norm_scale * nw[i];
            }
#pragma unroll
            for (int u = 0; u < kU; ++u) {
                const int i = base + u * blockDim.x;
                if (i < cols) stage[i] = v[u];
            }
        }
    }
    __syncthreads();
    const bool track = (flags & VINF_MK_FLAG_ARGMAX) != 0;
    float best_v = -INFINITY;
    int best_i = 0x7fffffff;
    const int r0 = row[VINF_MK_QMV_ROW_START], r1 = row[VINF_MK_QMV_ROW_END];
    const bool add = (flags & VINF_MK_FLAG_ADD) != 0;
#define VINF_MK_QMV_F(T)                                                                                       \
    case T:                                                                                                    \
        qmv_rows<T>(w, t.row_bytes, cols, r0, r1, stage, y, add, track, &best_v, &best_i, iq4nl,                  \
                    (T == kIQ3_S) ? iq3s : grid_source<T>(), warp_buf);                                        \
        break;
#define VINF_MK_QMV_Q(T)                                                                                           \
    case T:                                                                                                        \
        qmv_rows_q<T>(w, t.row_bytes, cols, r0, r1, xq, xd, xsum, y, add, track, &best_v, &best_i, iq4nl, iq3s, warp_buf); \
        break;
    switch (t.type) {
        VINF_MK_QMV_F(kF32)
        VINF_MK_QMV_F(kF16)
        VINF_MK_QMV_F(kQ2_K)
        VINF_MK_QMV_F(kIQ2_XXS)
        VINF_MK_QMV_F(kIQ2_XS)
        VINF_MK_QMV_F(kIQ2_S)
        VINF_MK_QMV_F(kIQ3_XXS)
        VINF_MK_QMV_Q(kQ8_0)
        VINF_MK_QMV_Q(kQ3_K)
        VINF_MK_QMV_Q(kQ4_K)
        VINF_MK_QMV_Q(kQ5_K)
        VINF_MK_QMV_Q(kQ6_K)
        VINF_MK_QMV_Q(kIQ4_NL)
        VINF_MK_QMV_Q(kIQ4_XS)
        VINF_MK_QMV_Q(kIQ3_S)
        default: break;
    }
#undef VINF_MK_QMV_F
#undef VINF_MK_QMV_Q
    if (track) {
        __shared__ float wv[kWarps];
        __shared__ int wi[kWarps];
        if (lane == 0) {
            wv[warp] = best_v;
            wi[warp] = best_i;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bv = wv[0];
            int bi = wi[0];
            for (int i = 1; i < kWarps; ++i) {
                if (wv[i] > bv || (wv[i] == bv && wi[i] < bi)) {
                    bv = wv[i];
                    bi = wi[i];
                }
            }
            a.partial_val[row[VINF_MK_QMV_PARTIAL]] = bv;
            a.partial_idx[row[VINF_MK_QMV_PARTIAL]] = bi;
        }
    }
}

// ---- LOAD --------------------------------------------------------------------------------

__device__ void op_load(const MkArgs &a, const int *row) {
    const uint8_t *src = a.tensors[row[VINF_MK_LOAD_TENSOR]].ptr;
    uint8_t *dst = a.slots[row[VINF_MK_LOAD_SLOT]];
    const size_t b0 = static_cast<size_t>(static_cast<unsigned>(row[VINF_MK_LOAD_BYTE_START]));
    const size_t b1 = static_cast<size_t>(static_cast<unsigned>(row[VINF_MK_LOAD_BYTE_END]));
    const size_t vec_end = b0 + ((b1 - b0) / 16) * 16;
    const uint4 *s4 = reinterpret_cast<const uint4 *>(src + b0);
    uint4 *d4 = reinterpret_cast<uint4 *>(dst + b0);
    const size_t n4 = (vec_end - b0) / 16;
    constexpr int kUnroll = 4;
    size_t i = threadIdx.x;
    for (; i + (kUnroll - 1) * blockDim.x < n4; i += kUnroll * blockDim.x) {
        uint4 tmp[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) tmp[u] = s4[i + u * blockDim.x];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) d4[i + u * blockDim.x] = tmp[u];
    }
    for (; i < n4; i += blockDim.x) d4[i] = s4[i];
    for (size_t b = vec_end + threadIdx.x; b < b1; b += blockDim.x) dst[b] = src[b];
}

// ---- ATTNHEAD ----------------------------------------------------------------------------

__device__ void rope_pairs(float *x, int rot, int pos, double base) {
    const int half = rot / 2;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        const double theta = static_cast<double>(pos) * pow(base, -2.0 * j / rot);
        const float c = static_cast<float>(cos(theta)), s = static_cast<float>(sin(theta));
        const float x0 = x[j], x1 = x[j + half];
        x[j] = x0 * c - x1 * s;
        x[j + half] = x1 * c + x0 * s;
    }
}

__device__ void norm_inplace(float *x, int n, const float *w, float eps) {
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local += x[i] * x[i];
    const float scale = rsqrtf(block_sum(local) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] = x[i] * scale * w[i];
    __syncthreads();
}

__device__ void op_attn_head(const MkArgs &a, const int *row, float *smem) {
    const MkParams &p = *a.params;
    const int hd = p.hd, h = row[VINF_MK_ATTNHEAD_HEAD];
    const int group = p.heads / p.kv_heads;
    const int g = h / group;
    const int pos = p.position;
    float *q = smem, *gate = q + hd, *k = gate + hd, *v = k + hd, *probs = v + hd;
    const float *q_raw = a.buffers[row[VINF_MK_ATTNHEAD_Q_RAW]];
    const float *kin = a.buffers[row[VINF_MK_ATTNHEAD_K]];
    const float *vin = a.buffers[row[VINF_MK_ATTNHEAD_V]];
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        q[d] = q_raw[h * 2 * hd + d];
        gate[d] = q_raw[h * 2 * hd + hd + d];
        k[d] = kin[g * hd + d];
        v[d] = vin[g * hd + d];
    }
    __syncthreads();
    norm_inplace(q, hd, reinterpret_cast<const float *>(a.tensors[row[VINF_MK_ATTNHEAD_Q_NORM]].ptr), p.eps);
    norm_inplace(k, hd, reinterpret_cast<const float *>(a.tensors[row[VINF_MK_ATTNHEAD_K_NORM]].ptr), p.eps);
    rope_pairs(q, p.rot, pos, p.freq_base);
    rope_pairs(k, p.rot, pos, p.freq_base);
    __syncthreads();
    float *kc = a.buffers[row[VINF_MK_ATTNHEAD_KC]] + static_cast<size_t>(g) * p.max_ctx * hd;
    float *vc = a.buffers[row[VINF_MK_ATTNHEAD_VC]] + static_cast<size_t>(g) * p.max_ctx * hd;
    if (h % group == 0) {
        for (int d = threadIdx.x; d < hd; d += blockDim.x) {
            kc[static_cast<size_t>(pos) * hd + d] = k[d];
            vc[static_cast<size_t>(pos) * hd + d] = v[d];
        }
    }
    // Positions < pos come from the cache (written by earlier launches); pos uses smem k/v.
    const float scale = rsqrtf(static_cast<float>(hd));
    float local_max = -INFINITY;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        const float *kt = t < pos ? kc + static_cast<size_t>(t) * hd : k;
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += q[d] * kt[d];
        probs[t] = dot * scale;
        local_max = fmaxf(local_max, probs[t]);
    }
    const float m = block_max(local_max);
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        probs[t] = expf(probs[t] - m);
        local_sum += probs[t];
    }
    const float denom = block_sum(local_sum);
    float *out = a.buffers[row[VINF_MK_ATTNHEAD_OUT]];
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = probs[pos] * v[d];
        for (int t = 0; t < pos; ++t) acc += probs[t] * vc[static_cast<size_t>(t) * hd + d];
        out[h * hd + d] = acc / denom * sigmoid_f(gate[d]);
    }
}

// ---- SSMGROUP ----------------------------------------------------------------------------

__device__ void op_ssm_group(const MkArgs &a, const int *row, float *smem) {
    const MkParams &p = *a.params;
    const int kh = row[VINF_MK_SSMGROUP_KEY_HEAD];
    const int kd = p.kd, vd = p.vd, K = p.conv_k;
    const int rep = p.value_heads / p.key_heads;
    const int key_dim = p.key_heads * kd;
    float *q = smem, *k = q + kd, *v = k + kd, *o = v + rep * vd;  // o: rep * vd
    const float *qkv = a.buffers[row[VINF_MK_SSMGROUP_QKV]];
    float *conv_state = a.buffers[row[VINF_MK_SSMGROUP_CONV_STATE]];
    const float *conv_w = reinterpret_cast<const float *>(a.tensors[row[VINF_MK_SSMGROUP_CONV_W]].ptr);
    // Causal conv + SiLU over this group's channels: q (kd), k (kd), v of each tiled value head (vd).
    const int nch = 2 * kd + rep * vd;
    for (int i = threadIdx.x; i < nch; i += blockDim.x) {
        int c;
        float *dst;
        if (i < kd) {
            c = kh * kd + i;
            dst = q + i;
        } else if (i < 2 * kd) {
            c = key_dim + kh * kd + (i - kd);
            dst = k + (i - kd);
        } else {
            const int j = i - 2 * kd, r = j / vd, vh = kh + r * p.key_heads;
            c = 2 * key_dim + vh * vd + (j % vd);
            dst = v + j;
        }
        float *st = conv_state + static_cast<size_t>(c) * K;
        for (int t = 0; t < K - 1; ++t) st[t] = st[t + 1];
        st[K - 1] = qkv[c];
        float acc = 0.0f;
        for (int t = 0; t < K; ++t) acc += st[t] * conv_w[static_cast<size_t>(c) * K + t];
        *dst = silu_f(acc);
    }
    __syncthreads();
    float qs = 0.0f, ks = 0.0f;
    for (int i = threadIdx.x; i < kd; i += blockDim.x) {
        qs += q[i] * q[i];
        ks += k[i] * k[i];
    }
    const float qn = sqrtf(block_sum(qs) + p.eps), kn = sqrtf(block_sum(ks) + p.eps);
    const float qscale = 1.0f / (qn * sqrtf(static_cast<float>(kd)));
    for (int i = threadIdx.x; i < kd; i += blockDim.x) {
        q[i] *= qscale;
        k[i] /= kn;
    }
    __syncthreads();
    const float *beta_raw = a.buffers[row[VINF_MK_SSMGROUP_BETA]];
    const float *alpha_raw = a.buffers[row[VINF_MK_SSMGROUP_ALPHA]];
    const float *z = a.buffers[row[VINF_MK_SSMGROUP_Z]];
    const float *ssm_a = reinterpret_cast<const float *>(a.tensors[row[VINF_MK_SSMGROUP_SSM_A]].ptr);
    const float *dt = reinterpret_cast<const float *>(a.tensors[row[VINF_MK_SSMGROUP_DT_BIAS]].ptr);
    const float *norm_w = reinterpret_cast<const float *>(a.tensors[row[VINF_MK_SSMGROUP_SSM_NORM]].ptr);
    float *state_all = a.buffers[row[VINF_MK_SSMGROUP_SSM_STATE]];
    float *out = a.buffers[row[VINF_MK_SSMGROUP_OUT]];
    // All tiled value heads of this key head in parallel: work item t -> (head r, column j).
    for (int t = threadIdx.x; t < rep * vd; t += blockDim.x) {
        const int r = t / vd, j = t % vd;
        const int vh = kh + r * p.key_heads;
        const float beta = sigmoid_f(beta_raw[vh]);
        const float xa = alpha_raw[vh] + dt[vh];
        const float decay = expf(ssm_a[vh] * (xa > 20.0f ? xa : log1pf(expf(xa))));
        float *S = state_all + static_cast<size_t>(vh) * kd * vd + j;
        float kv_mem = 0.0f;
#pragma unroll 8
        for (int i = 0; i < kd; ++i) kv_mem += S[static_cast<size_t>(i) * vd] * k[i];
        kv_mem *= decay;  // sum_i (S_ij * decay) k_i
        const float delta = (v[r * vd + j] - kv_mem) * beta;
        float acc = 0.0f;
#pragma unroll 8
        for (int i = 0; i < kd; ++i) {
            const float sij = S[static_cast<size_t>(i) * vd] * decay + k[i] * delta;
            S[static_cast<size_t>(i) * vd] = sij;
            acc += sij * q[i];
        }
        o[t] = acc;
    }
    __syncthreads();
    for (int r = 0; r < rep; ++r) {
        const int vh = kh + r * p.key_heads;
        float local = 0.0f;
        for (int j = threadIdx.x; j < vd; j += blockDim.x) local += o[r * vd + j] * o[r * vd + j];
        const float scale = rsqrtf(block_sum(local) / vd + p.eps);
        for (int j = threadIdx.x; j < vd; j += blockDim.x)
            out[vh * vd + j] = o[r * vd + j] * scale * norm_w[j] * silu_f(z[vh * vd + j]);
    }
}

// ---- ARGMAX ------------------------------------------------------------------------------

__device__ void op_argmax(const MkArgs &a, const int *row) {
    const int n = row[VINF_MK_ARGMAX_PARTIALS];
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = a.partial_val[i];
        const int idx = a.partial_idx[i];
        if (v > bv || (v == bv && idx < bi)) {
            bv = v;
            bi = idx;
        }
    }
    __shared__ float sv[kThreads];
    __shared__ int si[kThreads];
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float ov = sv[threadIdx.x + stride];
            const int oi = si[threadIdx.x + stride];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) a.params->out_token = si[0];
}

// ---- kernel ------------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads, 1) qwen_megakernel(MkArgs a) {
    extern __shared__ float smem[];
    __shared__ int8_t s_iq4nl[16];
    __shared__ uint8_t s_iq3s[512 * 4];
    __shared__ int s_ok;
    if (threadIdx.x < 16) s_iq4nl[threadIdx.x] = c_iq4nl_values[threadIdx.x];
    for (int i = threadIdx.x; i < 512 * 4; i += blockDim.x) s_iq3s[i] = c_iq3s_grid[i];
    __syncthreads();
    const int *queue = a.instr + static_cast<size_t>(blockIdx.x) * a.queue_len * VINF_MK_WORDS;
    for (int qi = 0; qi < a.queue_len; ++qi) {
        const int *row = queue + static_cast<size_t>(qi) * VINF_MK_WORDS;
        const int op = row[0];
        if (op == VINF_MK_OP_NOOP) break;  // queues are NoOp-padded at the tail only
        unsigned long long *tslot = a.timing ? a.timing + (static_cast<size_t>(blockIdx.x) * a.queue_len + qi) * 3 : nullptr;
        if (threadIdx.x == 0) {
            if (tslot) tslot[0] = global_ns();
            s_ok = wait_counter(a, row[VINF_MK_WAIT0_COUNTER], row[VINF_MK_WAIT0_TARGET], qi) &&
                   wait_counter(a, row[VINF_MK_WAIT1_COUNTER], row[VINF_MK_WAIT1_TARGET], qi);
            if (tslot) tslot[1] = global_ns();
        }
        __syncthreads();
        if (!s_ok) return;
        switch (op) {
            case VINF_MK_OP_QMV: op_qmv(a, row, smem, s_iq4nl, s_iq3s); break;
            case VINF_MK_OP_LOAD: op_load(a, row); break;
            case VINF_MK_OP_ATTNHEAD: op_attn_head(a, row, smem); break;
            case VINF_MK_OP_SSMGROUP: op_ssm_group(a, row, smem); break;
            case VINF_MK_OP_ARGMAX: op_argmax(a, row); break;
            default:
                if (threadIdx.x == 0 && atomicCAS(&a.params->error, 0, 2) == 0) {
                    a.params->error_block = blockIdx.x;
                    a.params->error_instr = qi;
                }
                return;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (tslot) tslot[2] = global_ns();
            signal_counter(a, row[VINF_MK_SIGNAL]);
            if (op == VINF_MK_OP_QMV) signal_counter(a, row[VINF_MK_QMV_CONSUMED_SIGNAL]);
        }
    }
}

// ---- host object -------------------------------------------------------------------------

PyObject *cuda_error(const char *context, cudaError_t err) {
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s", context, cudaGetErrorString(err));
    return nullptr;
}

struct CopyItem {
    const void *src;
    size_t nbytes;
    int slot;
    int wait_counter;
    int wait_target;
    int ready_counter;
};

struct MegakernelObject {
    PyObject_HEAD
    std::vector<CopyItem> *copy_plan;
    cudaStream_t copy_stream;
    cudaStream_t kernel_stream;
    cudaEvent_t start_event;
    int *instr;
    int num_blocks;
    int queue_len;
    MkTensor *tensors;
    float **buffers;
    uint8_t **slot_table;
    std::vector<uint8_t *> *slots;
    int *counters;
    int num_counters;
    float *partial_val;
    int *partial_idx;
    MkParams *params;
    unsigned long long *timing;
    MkParams host_params;
    size_t smem_bytes;
    bool configured;
};

void release(MegakernelObject *self) {
    cudaFree(self->instr);
    cudaFree(self->tensors);
    cudaFree(self->buffers);
    cudaFree(self->slot_table);
    if (self->slots != nullptr) {
        for (auto *s : *self->slots) cudaFree(s);
        self->slots->clear();
    }
    cudaFree(self->counters);
    cudaFree(self->partial_val);
    cudaFree(self->partial_idx);
    cudaFree(self->params);
    cudaFree(self->timing);
    self->timing = nullptr;
    if (self->copy_plan != nullptr) self->copy_plan->clear();
    self->instr = nullptr;
    self->tensors = nullptr;
    self->buffers = nullptr;
    self->slot_table = nullptr;
    self->counters = nullptr;
    self->partial_val = nullptr;
    self->partial_idx = nullptr;
    self->params = nullptr;
    self->configured = false;
}

int Mk_init(MegakernelObject *self, PyObject *args, PyObject *) {
    Py_buffer grid;
    if (!PyArg_ParseTuple(args, "y*", &grid)) return -1;
    if (grid.len != 512 * 4) {
        PyBuffer_Release(&grid);
        PyErr_SetString(PyExc_ValueError, "iq3s_grid must be 2048 bytes");
        return -1;
    }
    cudaError_t err = cudaMemcpyToSymbol(c_iq3s_grid, grid.buf, 512 * 4);
    PyBuffer_Release(&grid);
    if (err != cudaSuccess) {
        cuda_error("cudaMemcpyToSymbol", err);
        return -1;
    }
    if (self->slots == nullptr) self->slots = new std::vector<uint8_t *>();
    if (self->copy_plan == nullptr) self->copy_plan = new std::vector<CopyItem>();
    if (self->copy_stream == nullptr) {
        err = cudaStreamCreateWithFlags(&self->copy_stream, cudaStreamNonBlocking);
        if (err == cudaSuccess) err = cudaStreamCreateWithFlags(&self->kernel_stream, cudaStreamNonBlocking);
        if (err == cudaSuccess) err = cudaEventCreateWithFlags(&self->start_event, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            cuda_error("megakernel stream setup", err);
            return -1;
        }
    }
    return 0;
}

void Mk_dealloc(MegakernelObject *self) {
    release(self);
    delete self->slots;
    self->slots = nullptr;
    delete self->copy_plan;
    self->copy_plan = nullptr;
    if (self->copy_stream != nullptr) cudaStreamDestroy(self->copy_stream);
    if (self->kernel_stream != nullptr) cudaStreamDestroy(self->kernel_stream);
    if (self->start_event != nullptr) cudaEventDestroy(self->start_event);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *Mk_device_info(MegakernelObject *, PyObject *) {
    int dev = 0, sms = 0, coop = 0, max_smem = 0, blocks = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&blocks, cudaDevAttrClockRate, dev);  // kHz
    int mem_ops = 0;
    CUdevice cu_dev;
    if (cuDeviceGet(&cu_dev, dev) == CUDA_SUCCESS)
        cuDeviceGetAttribute(&mem_ops, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR, cu_dev);
    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i}", "sms", sms, "cooperative", coop, "max_smem_optin", max_smem,
                         "threads", kThreads, "clock_khz", blocks, "stream_mem_ops", mem_ops, "warps", kWarps,
                         "warp_buf_bytes", kWarpBufBytes);
}

// configure(instr_bytes, num_blocks, queue_len, tensors[(addr,type,rows,cols,row_bytes)], buffers[addr],
//           num_slots, slot_bytes, num_counters, num_partials, params{...}, smem_bytes)
PyObject *Mk_configure(MegakernelObject *self, PyObject *args) {
    Py_buffer instr;
    int num_blocks = 0, queue_len = 0, num_slots = 0, num_counters = 0, num_partials = 0;
    Py_ssize_t slot_bytes = 0, smem_bytes = 0;
    PyObject *tensors_obj, *buffers_obj, *params_obj, *copy_obj = nullptr;
    if (!PyArg_ParseTuple(args, "y*iiOOiniiOn|O", &instr, &num_blocks, &queue_len, &tensors_obj, &buffers_obj,
                          &num_slots, &slot_bytes, &num_counters, &num_partials, &params_obj, &smem_bytes, &copy_obj))
        return nullptr;
    release(self);
    if (copy_obj != nullptr && copy_obj != Py_None) {
        PyObject *cf = PySequence_Fast(copy_obj, "copy plan must be a sequence");
        if (cf == nullptr) {
            PyBuffer_Release(&instr);
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(cf); ++i) {
            unsigned long long src = 0;
            Py_ssize_t nbytes = 0;
            CopyItem item{};
            if (!PyArg_ParseTuple(PySequence_Fast_GET_ITEM(cf, i), "Kniiii", &src, &nbytes, &item.slot,
                                  &item.wait_counter, &item.wait_target, &item.ready_counter)) {
                Py_DECREF(cf);
                PyBuffer_Release(&instr);
                return nullptr;
            }
            item.src = reinterpret_cast<const void *>(static_cast<uintptr_t>(src));
            item.nbytes = static_cast<size_t>(nbytes);
            self->copy_plan->push_back(item);
        }
        Py_DECREF(cf);
    }
    if (instr.len != static_cast<Py_ssize_t>(num_blocks) * queue_len * VINF_MK_WORDS * sizeof(int)) {
        PyBuffer_Release(&instr);
        return PyErr_Format(PyExc_ValueError, "instruction bytes do not match num_blocks x queue_len x %d int32",
                            VINF_MK_WORDS);
    }
    std::vector<MkTensor> tensors;
    PyObject *tf = PySequence_Fast(tensors_obj, "tensors must be a sequence");
    if (tf == nullptr) {
        PyBuffer_Release(&instr);
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(tf); ++i) {
        unsigned long long addr = 0;
        MkTensor t{};
        if (!PyArg_ParseTuple(PySequence_Fast_GET_ITEM(tf, i), "Kiiii", &addr, &t.type, &t.rows, &t.cols, &t.row_bytes)) {
            Py_DECREF(tf);
            PyBuffer_Release(&instr);
            return nullptr;
        }
        t.ptr = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(addr));
        tensors.push_back(t);
    }
    Py_DECREF(tf);
    std::vector<float *> buffers;
    PyObject *bf = PySequence_Fast(buffers_obj, "buffers must be a sequence");
    if (bf == nullptr) {
        PyBuffer_Release(&instr);
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(bf); ++i) {
        buffers.push_back(reinterpret_cast<float *>(static_cast<uintptr_t>(PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(bf, i)))));
    }
    Py_DECREF(bf);
    if (PyErr_Occurred()) {
        PyBuffer_Release(&instr);
        return nullptr;
    }
    MkParams hp{};
#define VINF_GET_INT(field)                                                                \
    {                                                                                      \
        PyObject *v = PyDict_GetItemString(params_obj, #field);                            \
        if (v == nullptr) {                                                                \
            PyBuffer_Release(&instr);                                                      \
            return PyErr_Format(PyExc_KeyError, "params missing %s", #field);              \
        }                                                                                  \
        hp.field = static_cast<decltype(hp.field)>(PyFloat_Check(v) ? PyFloat_AsDouble(v) : PyLong_AsLongLong(v)); \
    }
    VINF_GET_INT(max_ctx) VINF_GET_INT(heads) VINF_GET_INT(kv_heads) VINF_GET_INT(hd) VINF_GET_INT(rot)
    VINF_GET_INT(freq_base) VINF_GET_INT(eps) VINF_GET_INT(key_heads) VINF_GET_INT(value_heads) VINF_GET_INT(kd)
    VINF_GET_INT(vd) VINF_GET_INT(conv_k) VINF_GET_INT(timeout_cycles)
#undef VINF_GET_INT
    self->host_params = hp;
    cudaError_t err = cudaMalloc(&self->instr, instr.len);
    if (err == cudaSuccess) err = cudaMemcpy(self->instr, instr.buf, instr.len, cudaMemcpyHostToDevice);
    PyBuffer_Release(&instr);
    const size_t nt = tensors.empty() ? 1 : tensors.size(), nb = buffers.empty() ? 1 : buffers.size();
    if (err == cudaSuccess) err = cudaMalloc(&self->tensors, nt * sizeof(MkTensor));
    if (err == cudaSuccess && !tensors.empty())
        err = cudaMemcpy(self->tensors, tensors.data(), tensors.size() * sizeof(MkTensor), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMalloc(&self->buffers, nb * sizeof(float *));
    if (err == cudaSuccess && !buffers.empty())
        err = cudaMemcpy(self->buffers, buffers.data(), buffers.size() * sizeof(float *), cudaMemcpyHostToDevice);
    std::vector<uint8_t *> slot_ptrs;
    for (int i = 0; i < num_slots && err == cudaSuccess; ++i) {
        uint8_t *s = nullptr;
        err = cudaMalloc(&s, slot_bytes + 16);  // aligned row loads may over-read 15 bytes
        self->slots->push_back(s);
        slot_ptrs.push_back(s);
    }
    if (err == cudaSuccess) err = cudaMalloc(&self->slot_table, (slot_ptrs.empty() ? 1 : slot_ptrs.size()) * sizeof(uint8_t *));
    if (err == cudaSuccess && !slot_ptrs.empty())
        err = cudaMemcpy(self->slot_table, slot_ptrs.data(), slot_ptrs.size() * sizeof(uint8_t *), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMalloc(&self->counters, (num_counters > 0 ? num_counters : 1) * sizeof(int));
    if (err == cudaSuccess) err = cudaMalloc(&self->partial_val, (num_partials > 0 ? num_partials : 1) * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(&self->partial_idx, (num_partials > 0 ? num_partials : 1) * sizeof(int));
    if (err == cudaSuccess) err = cudaMalloc(&self->params, sizeof(MkParams));
    if (err == cudaSuccess)
        err = cudaFuncSetAttribute(qwen_megakernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
    if (err != cudaSuccess) {
        release(self);
        return cuda_error("megakernel configure", err);
    }
    int per_sm = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, qwen_megakernel, kThreads, smem_bytes);
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess || per_sm * sms < num_blocks) {
        release(self);
        return PyErr_Format(PyExc_RuntimeError, "megakernel cannot co-schedule %d blocks (%d per SM x %d SMs, smem %zd)",
                            num_blocks, per_sm, sms, smem_bytes);
    }
    self->num_blocks = num_blocks;
    self->queue_len = queue_len;
    self->num_counters = num_counters;
    self->smem_bytes = static_cast<size_t>(smem_bytes);
    self->configured = true;
    Py_RETURN_NONE;
}

// run(position) -> (token, error, error_block, error_instr)
PyObject *Mk_run(MegakernelObject *self, PyObject *args) {
    int position = 0;
    if (!PyArg_ParseTuple(args, "i", &position)) return nullptr;
    if (!self->configured) return PyErr_Format(PyExc_RuntimeError, "megakernel is not configured");
    if (position < 0 || position >= self->host_params.max_ctx)
        return PyErr_Format(PyExc_ValueError, "position %d outside max_ctx %d", position, self->host_params.max_ctx);
    MkParams hp = self->host_params;
    hp.position = position;
    hp.out_token = -1;
    hp.error = 0;
    hp.error_block = -1;
    hp.error_instr = -1;
    cudaError_t err = cudaMemcpy(self->params, &hp, sizeof(MkParams), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemset(self->counters, 0, (self->num_counters > 0 ? self->num_counters : 1) * sizeof(int));
    if (err == cudaSuccess) err = cudaDeviceSynchronize();  // counters/params visible before either stream starts
    MkArgs a{self->instr, self->queue_len, self->tensors, self->buffers, self->slot_table, self->counters,
             self->partial_val, self->partial_idx, self->params, self->timing};
    void *kargs[] = {&a};
    CUresult cu = CUDA_SUCCESS;
    const CUdeviceptr counters = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(self->counters));
    Py_BEGIN_ALLOW_THREADS
    for (const CopyItem &item : *self->copy_plan) {
        if (err != cudaSuccess || cu != CUDA_SUCCESS) break;
        if (item.wait_counter >= 0)
            cu = cuStreamWaitValue32(reinterpret_cast<CUstream>(self->copy_stream), counters + item.wait_counter * sizeof(int),
                                     static_cast<cuuint32_t>(item.wait_target), CU_STREAM_WAIT_VALUE_GEQ);
        if (cu == CUDA_SUCCESS)
            err = cudaMemcpyAsync((*self->slots)[item.slot], item.src, item.nbytes, cudaMemcpyHostToDevice, self->copy_stream);
        if (err == cudaSuccess && cu == CUDA_SUCCESS)
            cu = cuStreamWriteValue32(reinterpret_cast<CUstream>(self->copy_stream), counters + item.ready_counter * sizeof(int),
                                      1, CU_STREAM_WRITE_VALUE_DEFAULT);
    }
    if (err == cudaSuccess && cu == CUDA_SUCCESS)
        err = cudaLaunchCooperativeKernel(reinterpret_cast<void *>(qwen_megakernel), dim3(self->num_blocks), dim3(kThreads),
                                          kargs, self->smem_bytes, self->kernel_stream);
    if (err == cudaSuccess && cu == CUDA_SUCCESS) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(&hp, self->params, sizeof(MkParams), cudaMemcpyDeviceToHost);
    Py_END_ALLOW_THREADS
    if (cu != CUDA_SUCCESS) {
        const char *msg = nullptr;
        cuGetErrorString(cu, &msg);
        return PyErr_Format(PyExc_RuntimeError, "megakernel copy plan failed: %s", msg ? msg : "unknown");
    }
    if (err != cudaSuccess) return cuda_error("megakernel run", err);
    return Py_BuildValue("(iiii)", hp.out_token, hp.error, hp.error_block, hp.error_instr);
}

PyObject *Mk_counters(MegakernelObject *self, PyObject *) {
    std::vector<int> host(self->num_counters > 0 ? self->num_counters : 0);
    if (!host.empty()) {
        cudaError_t err = cudaMemcpy(host.data(), self->counters, host.size() * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return cuda_error("read counters", err);
    }
    PyObject *list = PyList_New(static_cast<Py_ssize_t>(host.size()));
    for (size_t i = 0; i < host.size(); ++i) PyList_SET_ITEM(list, i, PyLong_FromLong(host[i]));
    return list;
}

// set_timing(enabled): allocate/free the per-instruction timing buffer (debug; adds overhead).
PyObject *Mk_set_timing(MegakernelObject *self, PyObject *args) {
    int enabled = 0;
    if (!PyArg_ParseTuple(args, "p", &enabled)) return nullptr;
    cudaFree(self->timing);
    self->timing = nullptr;
    if (enabled && self->configured) {
        const size_t n = static_cast<size_t>(self->num_blocks) * self->queue_len * 3;
        cudaError_t err = cudaMalloc(&self->timing, n * sizeof(unsigned long long));
        if (err == cudaSuccess) err = cudaMemset(self->timing, 0, n * sizeof(unsigned long long));
        if (err != cudaSuccess) return cuda_error("timing buffer", err);
    }
    Py_RETURN_NONE;
}

// timing() -> bytes of uint64 [blocks][queue_len][3] from the last run.
PyObject *Mk_timing(MegakernelObject *self, PyObject *) {
    if (self->timing == nullptr) Py_RETURN_NONE;
    const size_t n = static_cast<size_t>(self->num_blocks) * self->queue_len * 3;
    PyObject *out = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(n * sizeof(unsigned long long)));
    if (out == nullptr) return nullptr;
    cudaError_t err = cudaMemcpy(PyBytes_AS_STRING(out), self->timing, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        Py_DECREF(out);
        return cuda_error("timing read", err);
    }
    return out;
}

PyMethodDef Mk_methods[] = {
    {"set_timing", reinterpret_cast<PyCFunction>(Mk_set_timing), METH_VARARGS, "Enable per-instruction timing."},
    {"timing", reinterpret_cast<PyCFunction>(Mk_timing), METH_NOARGS, "Per-instruction timestamps (ns) of the last run."},
    {"device_info", reinterpret_cast<PyCFunction>(Mk_device_info), METH_NOARGS, "SM count, cooperative support, smem limits."},
    {"configure", reinterpret_cast<PyCFunction>(Mk_configure), METH_VARARGS, "Upload instruction queues and tables."},
    {"run", reinterpret_cast<PyCFunction>(Mk_run), METH_VARARGS, "run(position) -> (token, error, block, instr)."},
    {"counters", reinterpret_cast<PyCFunction>(Mk_counters), METH_NOARGS, "Counter values after the last run (debug)."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject MegakernelType = {PyVarObject_HEAD_INIT(nullptr, 0)};
PyModuleDef Module = {PyModuleDef_HEAD_INIT, "_cuda_qwen_megakernel", "qwen35 fused decode megakernel.", -1, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__cuda_qwen_megakernel(void) {
    MegakernelType.tp_name = "vinf._cuda_qwen_megakernel.Megakernel";
    MegakernelType.tp_basicsize = sizeof(MegakernelObject);
    MegakernelType.tp_flags = Py_TPFLAGS_DEFAULT;
    MegakernelType.tp_new = PyType_GenericNew;
    MegakernelType.tp_init = reinterpret_cast<initproc>(Mk_init);
    MegakernelType.tp_dealloc = reinterpret_cast<destructor>(Mk_dealloc);
    MegakernelType.tp_methods = Mk_methods;
    if (PyType_Ready(&MegakernelType) < 0) return nullptr;
    PyObject *module = PyModule_Create(&Module);
    if (module == nullptr) return nullptr;
    Py_INCREF(&MegakernelType);
    if (PyModule_AddObject(module, "Megakernel", reinterpret_cast<PyObject *>(&MegakernelType)) < 0) {
        Py_DECREF(&MegakernelType);
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}
