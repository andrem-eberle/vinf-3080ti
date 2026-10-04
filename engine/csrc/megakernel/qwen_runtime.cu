// Persistent CUDA runtime for qwen35 decode.
//
// Owns raw GGUF tensor blocks (VRAM-resident, or pinned host memory streamed into
// a device staging buffer on use), named float32 device buffers (activations, KV
// cache, SSM state), and the per-op kernels of the qwen35 hybrid decoder. Python
// orchestrates the op sequence; numeric work and activations stay on the GPU.
// Quantized row layouts mirror the ggml block formats verified by vinf.gguf.dequant.
//
// Kernel contract (vinf_qmatvec_kernel):
//   grid = ceil(rows / 8), block = 8 warps; warp w of block b owns row 8b + w.
//   Lanes stride over 8-element chunks, dequantize, dot with x, warp-reduce.
//   Output ownership: lane 0 of the owning warp writes y[row].
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "config_3080ti.cuh"
#include "math_utils.cuh"
#include "qwen_quant.cuh"

namespace {


// y[g*n + i] = x[g*n + i] * rsqrt(mean_g(x^2) + eps) * w[i]; grid = groups. In-place safe.
__global__ void rmsnorm_kernel(const float *x, const float *w, float *y, int n, float eps) {
    const float *xg = x + static_cast<size_t>(blockIdx.x) * n;
    float *yg = y + static_cast<size_t>(blockIdx.x) * n;
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local += xg[i] * xg[i];
    const float scale = rsqrtf(block_sum(local) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yg[i] = xg[i] * scale * w[i];
}

// Gated RMSNorm per head: y = rmsnorm(x) * w * silu(z); grid = heads.
__global__ void gated_rmsnorm_kernel(const float *x, const float *w, const float *z, float *y, int n, float eps) {
    const size_t base = static_cast<size_t>(blockIdx.x) * n;
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local += x[base + i] * x[base + i];
    const float scale = rsqrtf(block_sum(local) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[base + i] = x[base + i] * scale * w[i] * silu_f(z[base + i]);
}

__global__ void add_kernel(const float *a, const float *b, float *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void silu_mul_kernel(const float *gate, const float *up, float *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = silu_f(gate[i]) * up[i];
}

__global__ void sigmoid_mul_kernel(const float *x, const float *gate, float *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * sigmoid_f(gate[i]);
}

// qwen35 gated query: raw rows per head are [query(hd), gate(hd)].
__global__ void split_gated_q_kernel(const float *raw, float *q, float *gate, int heads, int hd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * hd) return;
    const int h = i / hd, d = i % hd;
    q[i] = raw[h * 2 * hd + d];
    gate[i] = raw[h * 2 * hd + hd + d];
}

// NeoX RoPE over the first `rot` dims of each head; remaining dims untouched.
// Token-batched kernels: activations are row-major [ntok][...]; token t sits at position pos0 + t.
// With ntok = 1 every kernel computes exactly the single-token result.

// NeoX RoPE over the first `rot` dims of each head; remaining dims untouched.
__global__ void rope_neox_kernel(float *x, int ntok, int heads, int hd, int rot, int pos0, double freq_base) {
    const int half = rot / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntok * heads * half) return;
    const int t = i / (heads * half), h = (i / half) % heads, j = i % half;
    const double theta = static_cast<double>(pos0 + t) * pow(freq_base, -2.0 * j / rot);
    const float c = static_cast<float>(cos(theta)), s = static_cast<float>(sin(theta));
    float *head = x + (static_cast<size_t>(t) * heads + h) * hd;
    const float x0 = head[j], x1 = head[j + half];
    head[j] = x0 * c - x1 * s;
    head[j + half] = x1 * c + x0 * s;
}

// KV cache element access: fp32 or fp16 caches (fp16 halves the cache; attention math stays fp32).
__device__ __forceinline__ float kv_load(const float *p) { return *p; }
__device__ __forceinline__ float kv_load(const __half *p) { return __half2float(*p); }
__device__ __forceinline__ void kv_store(float *p, float v) { *p = v; }
__device__ __forceinline__ void kv_store(__half *p, float v) { *p = __float2half_rn(v); }

// KV cache addressing. Contiguous (pt == nullptr): [kv_head][max_seq][hd]. Paged: a pool of pages
// [page][kv_head][page_size][hd]; pt is the sequence's page table (pt[pos / page_size] = page id).
struct KvAddr {
    const int *pt;
    int max_seq;
    int page;
    int kv_heads;
    __device__ __forceinline__ size_t row(int kvh, int pos) const {
        if (pt == nullptr) return static_cast<size_t>(kvh) * max_seq + pos;
        const int pg = pt[pos / page];
        return (static_cast<size_t>(pg) * kv_heads + kvh) * page + pos % page;
    }
};

inline KvAddr contiguous_kv(int max_seq) { return KvAddr{nullptr, max_seq, 0, 0}; }

template <typename T>
__global__ void kv_append_kernel(const float *k, const float *v, T *kc, T *vc, int ntok, int kv_heads,
                                 KvAddr A, int hd, int pos0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntok * kv_heads * hd) return;
    const int t = i / (kv_heads * hd), h = (i / hd) % kv_heads, d = i % hd;
    const size_t dst = A.row(h, pos0 + t) * hd + d;
    kv_store(kc + dst, k[i]);
    kv_store(vc + dst, v[i]);
}

// GQA attention: query token t (grid.y) attends to positions [start, seq_len0 + t) (causal) or
// [start, seq_len0 + ntok - 1) for every token (bidirectional block, DFlash drafting).
// window > 0 limits each query to its last `window` positions (sliding-window attention).
// grid = (heads, ntok); dynamic smem = (seq_len0 + ntok - 1) floats.
template <typename T>
__global__ void attention_kernel(const float *q, const T *kc, const T *vc, float *out, int heads, int kv_heads,
                                 int hd, KvAddr A, int seq_len0, float scale, int bidirectional, int window) {
    extern __shared__ float probs_all[];
    const int h = blockIdx.x, t = blockIdx.y;
    const int seq_len = bidirectional ? seq_len0 + static_cast<int>(gridDim.y) - 1 : seq_len0 + t;
    const int qpos1 = seq_len0 + t;  // query position + 1
    const int start = window > 0 && qpos1 > window ? qpos1 - window : 0;
    float *probs = probs_all - start;  // index by absolute position
    const int kvh = h / (heads / kv_heads);
    const float *qh = q + (static_cast<size_t>(t) * heads + h) * hd;
    float local_max = -INFINITY;
    for (int p = start + threadIdx.x; p < seq_len; p += blockDim.x) {
        const T *kp = kc + A.row(kvh, p) * hd;
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qh[d] * kv_load(kp + d);
        probs[p] = dot * scale;
        local_max = fmaxf(local_max, probs[p]);
    }
    const float m = block_max(local_max);
    float local_sum = 0.0f;
    for (int p = start + threadIdx.x; p < seq_len; p += blockDim.x) {
        probs[p] = expf(probs[p] - m);
        local_sum += probs[p];
    }
    const float denom = block_sum(local_sum);
    float *o = out + (static_cast<size_t>(t) * heads + h) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (int p = start; p < seq_len; ++p) acc += probs[p] * kv_load(vc + A.row(kvh, p) * hd + d);
        o[d] = acc / denom;
    }
}

// Tiled (flash-style) GQA attention for prompt passes and long contexts: a block takes kAttnTQ query
// tokens of one head and walks the keys in kAttnTK-position tiles staged in shared memory with an
// online softmax, so K/V are read once per query tile instead of once per query token, and shared
// memory does not grow with the context. Same visibility rules as attention_kernel.
// grid = (heads, ceil(ntok / kAttnTQ)); block = hd threads (thread d owns output dimension d).
constexpr int kAttnTQ = 16;
constexpr int kAttnTK = 32;

__host__ __device__ constexpr size_t attn_tiled_smem(int hd) {
    return (static_cast<size_t>(kAttnTQ) * (hd + 1) + static_cast<size_t>(kAttnTK) * (hd + 1) * 2 +
            kAttnTQ * kAttnTK + 3 * kAttnTQ) * sizeof(float);
}

template <typename T>
__global__ void attention_tiled_kernel(const float *q, const T *kc, const T *vc, float *out, int heads,
                                       int kv_heads, int hd, KvAddr A, int seq_len0, int ntok, float scale,
                                       int bidirectional, int window) {
    extern __shared__ float smem[];
    const int ld = hd + 1;
    float *qs = smem;                          // [TQ][hd+1]
    float *ks = qs + kAttnTQ * ld;             // [TK][hd+1]
    float *vs = ks + kAttnTK * ld;             // [TK][hd+1]
    float *sc = vs + kAttnTK * ld;             // [TQ][TK] scores -> probabilities
    float *row_m = sc + kAttnTQ * kAttnTK;     // running max per query
    float *row_l = row_m + kAttnTQ;            // running denominator per query
    float *row_a = row_l + kAttnTQ;            // rescale factor of the current tile
    const int h = blockIdx.x, t0 = blockIdx.y * kAttnTQ, tid = threadIdx.x, nthr = blockDim.x;
    const int nq = min(kAttnTQ, ntok - t0);
    const int kvh = h / (heads / kv_heads);
    for (int i = tid; i < kAttnTQ * hd; i += nthr) {
        const int r = i / hd, d = i % hd;
        qs[r * ld + d] = r < nq ? q[(static_cast<size_t>(t0 + r) * heads + h) * hd + d] * scale : 0.0f;
    }
    if (tid < kAttnTQ) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
    }
    // Key range of the tile: the union of the queries' visible ranges.
    auto q_end = [&](int r) { return bidirectional ? seq_len0 + ntok - 1 : seq_len0 + t0 + r; };
    auto q_start = [&](int r) {
        const int qpos1 = seq_len0 + t0 + r;
        return window > 0 && qpos1 > window ? qpos1 - window : 0;
    };
    const int kbeg = q_start(0), kend = q_end(nq - 1);
    float acc[kAttnTQ];
#pragma unroll
    for (int r = 0; r < kAttnTQ; ++r) acc[r] = 0.0f;
    __syncthreads();
    for (int k0 = kbeg - kbeg % kAttnTK; k0 < kend; k0 += kAttnTK) {
        const int nk = min(kAttnTK, kend - k0);
        for (int i = tid; i < kAttnTK * hd; i += nthr) {
            const int r = i / hd, d = i % hd;
            const bool in = r < nk;
            ks[r * ld + d] = in ? kv_load(kc + A.row(kvh, k0 + r) * hd + d) : 0.0f;
            vs[r * ld + d] = in ? kv_load(vc + A.row(kvh, k0 + r) * hd + d) : 0.0f;
        }
        __syncthreads();
        for (int i = tid; i < kAttnTQ * kAttnTK; i += nthr) {
            const int r = i / kAttnTK, c = i % kAttnTK, p = k0 + c;
            float dot = -INFINITY;
            if (r < nq && c < nk && p >= q_start(r) && p < q_end(r)) {
                dot = 0.0f;
                const float *qr = qs + r * ld, *kr = ks + c * ld;
                for (int d = 0; d < hd; ++d) dot += qr[d] * kr[d];
            }
            sc[i] = dot;
        }
        __syncthreads();
        if (tid < kAttnTQ) {
            float m = row_m[tid];
            for (int c = 0; c < kAttnTK; ++c) m = fmaxf(m, sc[tid * kAttnTK + c]);
            const float a = m == -INFINITY ? 1.0f : expf(row_m[tid] - m);
            float l = row_l[tid] * a;
            for (int c = 0; c < kAttnTK; ++c) {
                const float e = sc[tid * kAttnTK + c] == -INFINITY ? 0.0f : expf(sc[tid * kAttnTK + c] - m);
                sc[tid * kAttnTK + c] = e;
                l += e;
            }
            row_m[tid] = m;
            row_l[tid] = l;
            row_a[tid] = a;
        }
        __syncthreads();
        for (int d = tid; d < hd; d += nthr) {  // nthr == hd in practice
#pragma unroll
            for (int r = 0; r < kAttnTQ; ++r) {
                float sum = 0.0f;
                for (int c = 0; c < nk; ++c) sum += sc[r * kAttnTK + c] * vs[c * ld + d];
                acc[r] = acc[r] * row_a[r] + sum;
            }
        }
        __syncthreads();
    }
    for (int d = tid; d < hd; d += nthr) {
        for (int r = 0; r < nq; ++r) {
            const float l = row_l[r];
            out[(static_cast<size_t>(t0 + r) * heads + h) * hd + d] = l > 0.0f ? acc[r] / l : 0.0f;
        }
    }
}

// Tensor-core attention for prompt passes (flash-style). A block takes kTcQ query tokens of one head and
// walks the keys in kTcK-position tiles: S = Q K^T and O += P V run as 16x16x16 fp16 MMAs with fp32
// accumulation; the online softmax and the output accumulator stay in fp32 shared memory.
// Visibility rules match attention_kernel. head_dim must be a multiple of 16 and at most 256.
// grid = (heads, ceil(ntok / kTcQ)), block = 32 * kTcWarps.
constexpr int kTcQ = 32, kTcK = 32, kTcWarps = 4;

__host__ __device__ constexpr int tc_ld_h(int hd) { return hd + 8; }  // fp16 row stride (keeps 32B fragment alignment)
__host__ __device__ constexpr int tc_ld_o(int hd) { return hd + 8; }  // fp32 row stride
__host__ __device__ constexpr size_t attn_tc_smem(int hd) {
    return static_cast<size_t>(kTcQ) * tc_ld_h(hd) * 2      // Q fp16
           + static_cast<size_t>(kTcK) * tc_ld_h(hd) * 2    // K, then V, fp16
           + static_cast<size_t>(kTcQ) * (kTcK + 4) * 4     // S fp32
           + static_cast<size_t>(kTcQ) * (kTcK + 8) * 2     // P fp16
           + static_cast<size_t>(kTcQ) * tc_ld_o(hd) * 4    // O fp32
           + 3 * kTcQ * 4;                                  // m, l, alpha
}

template <typename T>
__global__ void __launch_bounds__(32 * kTcWarps) attention_tc_kernel(
    const float *q, const T *kc, const T *vc, float *out, int heads, int kv_heads, int hd, KvAddr A, int seq_len0,
    int ntok, float scale, int bidirectional, int window) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char tc_smem[];
    const int ldh = tc_ld_h(hd), ldo = tc_ld_o(hd), lds = kTcK + 4, ldp = kTcK + 8;
    __half *qs = reinterpret_cast<__half *>(tc_smem);
    __half *kvs = qs + kTcQ * ldh;
    float *ss = reinterpret_cast<float *>(kvs + kTcK * ldh);
    __half *ps = reinterpret_cast<__half *>(ss + kTcQ * lds);
    float *os = reinterpret_cast<float *>(ps + kTcQ * ldp);
    float *row_m = os + kTcQ * ldo, *row_l = row_m + kTcQ, *row_a = row_l + kTcQ;
    const int h = blockIdx.x, t0 = blockIdx.y * kTcQ, tid = threadIdx.x, nthr = blockDim.x, warp = tid / 32;
    const int nq = min(kTcQ, ntok - t0);
    const int kvh = h / (heads / kv_heads);
    for (int i = tid; i < kTcQ * hd; i += nthr) {
        const int r = i / hd, d = i % hd;
        qs[r * ldh + d] = __float2half_rn(r < nq ? q[(static_cast<size_t>(t0 + r) * heads + h) * hd + d] * scale : 0.0f);
        os[r * ldo + d] = 0.0f;
    }
    if (tid < kTcQ) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
    }
    auto q_end = [&](int r) { return bidirectional ? seq_len0 + ntok - 1 : seq_len0 + t0 + r; };
    auto q_start = [&](int r) {
        const int qpos1 = seq_len0 + t0 + r;
        return window > 0 && qpos1 > window ? qpos1 - window : 0;
    };
    const int kbeg = q_start(0), kend = q_end(nq - 1);
    __syncthreads();
    for (int k0 = kbeg - kbeg % kTcK; k0 < kend; k0 += kTcK) {
        const int nk = min(kTcK, kend - k0);
        for (int i = tid; i < kTcK * hd; i += nthr) {
            const int r = i / hd, d = i % hd;
            kvs[r * ldh + d] = __float2half_rn(r < nk ? kv_load(kc + A.row(kvh, k0 + r) * hd + d) : 0.0f);
        }
        __syncthreads();
        {  // S = Q K^T: 2 x 2 fragments of 16 x 16, one per warp
            const int fr = (warp / 2) * 16, fc = (warp % 2) * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int d = 0; d < hd; d += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, qs + fr * ldh + d, ldh);
                wmma::load_matrix_sync(b, kvs + fc * ldh + d, ldh);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(ss + fr * lds + fc, acc, lds, wmma::mem_row_major);
        }
        __syncthreads();
        // V tile replaces K; the softmax update runs alongside.
        for (int i = tid; i < kTcK * hd; i += nthr) {
            const int r = i / hd, d = i % hd;
            kvs[r * ldh + d] = __float2half_rn(r < nk ? kv_load(vc + A.row(kvh, k0 + r) * hd + d) : 0.0f);
        }
        if (tid < kTcQ) {
            const int r = tid;
            const int lo = q_start(r), hi = q_end(r);
            float m = row_m[r];
            for (int c = 0; c < kTcK; ++c) {
                const int pos = k0 + c;
                if (r < nq && c < nk && pos >= lo && pos < hi) m = fmaxf(m, ss[r * lds + c]);
            }
            const float a = m == -INFINITY ? 1.0f : expf(row_m[r] - m);
            float l = row_l[r] * a;
            for (int c = 0; c < kTcK; ++c) {
                const int pos = k0 + c;
                const bool ok = r < nq && c < nk && pos >= lo && pos < hi;
                const float e = ok ? expf(ss[r * lds + c] - m) : 0.0f;
                ps[r * ldp + c] = __float2half_rn(e);
                l += e;
            }
            row_m[r] = m;
            row_l[r] = l;
            row_a[r] = a;
        }
        __syncthreads();
        for (int i = tid; i < kTcQ * hd; i += nthr) {
            const int r = i / hd, d = i % hd;
            os[r * ldo + d] *= row_a[r];
        }
        __syncthreads();
        {  // O += P V: rows (warp / 2) * 16, column fragments split between warp pairs
            const int fr = (warp / 2) * 16;
            const int nfrag = hd / 16;
            for (int f = warp % 2; f < nfrag; f += 2) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::load_matrix_sync(acc, os + fr * ldo + f * 16, ldo, wmma::mem_row_major);
#pragma unroll
                for (int kk = 0; kk < kTcK; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                    wmma::load_matrix_sync(a, ps + fr * ldp + kk, ldp);
                    wmma::load_matrix_sync(b, kvs + kk * ldh + f * 16, ldh);
                    wmma::mma_sync(acc, a, b, acc);
                }
                wmma::store_matrix_sync(os + fr * ldo + f * 16, acc, ldo, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }
    for (int i = tid; i < nq * hd; i += nthr) {
        const int r = i / hd, d = i % hd;
        const float l = row_l[r];
        out[(static_cast<size_t>(t0 + r) * heads + h) * hd + d] = l > 0.0f ? os[r * ldo + d] / l : 0.0f;
    }
}

// Causal depthwise conv + SiLU stepping through ntok tokens. state: [channel][K] (last K inputs).
// snap (optional): state after each token, [ntok][channel][K].
// write_state = 0 leaves the state untouched (speculative verification; see the commit replay).
constexpr int kConvMaxK = 8;
__global__ void conv_update_kernel(const float *x, float *state, const float *w, float *out, int channels, int K,
                                   int ntok, float *snap, int write_state) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float *gst = state + static_cast<size_t>(c) * K;
    float st[kConvMaxK];
    for (int k = 0; k < K; ++k) st[k] = gst[k];
    for (int t = 0; t < ntok; ++t) {
        for (int k = 0; k < K - 1; ++k) st[k] = st[k + 1];
        st[K - 1] = x[static_cast<size_t>(t) * channels + c];
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += st[k] * w[static_cast<size_t>(c) * K + k];
        out[static_cast<size_t>(t) * channels + c] = silu_f(acc);
        if (snap != nullptr && t < ntok - 1) {
            float *dst = snap + (static_cast<size_t>(t) * channels + c) * K;
            for (int k = 0; k < K; ++k) dst[k] = st[k];
        }
    }
    if (write_state)
        for (int k = 0; k < K; ++k) gst[k] = st[k];
}

// Gated delta rule with the state in registers. Every state column j evolves independently given the
// token's normalized q/k, so a block takes 32 columns of one value head (grid = value_heads x vd/32) and
// kGdP warps split the kd rows: thread (warp p, lane c) holds rows [p*R, (p+1)*R) of column 32*by + c
// (R = kd / kGdP) for the whole pass. The state is read and written once per call instead of per token;
// per token the block needs three barriers (q/k norms, k.S reduction, q.S reduction).
constexpr int kGdP = 4;

template <int R>
__global__ void __launch_bounds__(32 * kGdP) gated_delta_reg_kernel(
    const float *conv_out, const float *beta_raw, const float *alpha_raw, const float *ssm_a, const float *dt_bias,
    float *state, float *out, int key_heads, int value_heads, int vd, float eps, int head_order, int ntok, float *snap,
    int write_state) {
    constexpr int kd = R * kGdP;
    __shared__ float qraw[kd], kraw[kd];
    __shared__ float red_n[2][kGdP], red_k[kGdP][32], red_o[kGdP][32];
    const int h = blockIdx.x, lane = threadIdx.x % 32, p = threadIdx.x / 32;
    const int j = blockIdx.y * 32 + lane;
    const int kh = head_order == 0 ? h / (value_heads / key_heads) : h % key_heads;
    const int conv_dim = 2 * key_heads * kd + value_heads * vd;
    float *S = state + static_cast<size_t>(h) * kd * vd;
    float st[R];
#pragma unroll
    for (int r = 0; r < R; ++r) st[r] = S[static_cast<size_t>(p * R + r) * vd + j];
    const float a_h = ssm_a[h], dt_h = dt_bias[h];
    for (int t = 0; t < ntok; ++t) {
        const float *row = conv_out + static_cast<size_t>(t) * conv_dim;
        const float *qsrc = row + kh * kd;
        const float *ksrc = row + key_heads * kd + kh * kd;
        float qs = 0.0f, ks = 0.0f;
        for (int i = threadIdx.x; i < kd; i += blockDim.x) {
            const float qv = qsrc[i], kv = ksrc[i];
            qraw[i] = qv;
            kraw[i] = kv;
            qs += qv * qv;
            ks += kv * kv;
        }
        qs = warp_sum(qs);
        ks = warp_sum(ks);
        if (lane == 0) {
            red_n[0][p] = qs;
            red_n[1][p] = ks;
        }
        const float vj = row[2 * key_heads * kd + h * vd + j];
        const float beta = sigmoid_f(beta_raw[static_cast<size_t>(t) * value_heads + h]);
        const float xa = alpha_raw[static_cast<size_t>(t) * value_heads + h] + dt_h;
        const float decay = expf(a_h * (xa > 20.0f ? xa : log1pf(expf(xa))));
        __syncthreads();
        float qsum = 0.0f, ksum = 0.0f;
#pragma unroll
        for (int w = 0; w < kGdP; ++w) {
            qsum += red_n[0][w];
            ksum += red_n[1][w];
        }
        const float qscale = 1.0f / (sqrtf(qsum + eps) * sqrtf(static_cast<float>(kd)));
        const float kscale = 1.0f / sqrtf(ksum + eps);
        float kv_mem = 0.0f;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            st[r] *= decay;
            kv_mem += st[r] * (kraw[p * R + r] * kscale);
        }
        red_k[p][lane] = kv_mem;
        __syncthreads();
        float kv_all = 0.0f;
#pragma unroll
        for (int w = 0; w < kGdP; ++w) kv_all += red_k[w][lane];
        const float delta = (vj - kv_all) * beta;
        float o = 0.0f;
        float *snap_t = (snap != nullptr && t < ntok - 1) ? snap + (static_cast<size_t>(t) * value_heads + h) * kd * vd : nullptr;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            st[r] += (kraw[p * R + r] * kscale) * delta;
            o += st[r] * (qraw[p * R + r] * qscale);
            if (snap_t != nullptr) snap_t[static_cast<size_t>(p * R + r) * vd + j] = st[r];
        }
        red_o[p][lane] = o;
        __syncthreads();
        if (p == 0) {
            float o_all = 0.0f;
#pragma unroll
            for (int w = 0; w < kGdP; ++w) o_all += red_o[w][lane];
            out[static_cast<size_t>(t) * value_heads * vd + h * vd + j] = o_all;
        }
    }
    if (write_state) {
#pragma unroll
        for (int r = 0; r < R; ++r) S[static_cast<size_t>(p * R + r) * vd + j] = st[r];
    }
}

// Gated delta rule stepping through ntok tokens. grid = value_heads, block = vd threads (thread j owns
// state column j). conv_out rows = [q(key_dim) | k(key_dim) | v(value_dim)]; state layout [vh][kd][vd].
// head_order 0: value head h uses key head h / (vh/kh) (grouped); 1: h % kh (tiled).
// snap (optional): state after each token, [ntok][vh][kd][vd].
__global__ void gated_delta_kernel(const float *conv_out, const float *beta_raw, const float *alpha_raw,
                                   const float *ssm_a, const float *dt_bias, float *state, float *out, int key_heads,
                                   int value_heads, int kd, int vd, float eps, int head_order, int ntok, float *snap) {
    extern __shared__ float qk[];  // [kd] q, [kd] k
    const int h = blockIdx.x;
    const int kh = head_order == 0 ? h / (value_heads / key_heads) : h % key_heads;
    const int conv_dim = 2 * key_heads * kd + value_heads * vd;
    float *S = state + static_cast<size_t>(h) * kd * vd;
    for (int t = 0; t < ntok; ++t) {
        const float *row = conv_out + static_cast<size_t>(t) * conv_dim;
        const float *qsrc = row + kh * kd;
        const float *ksrc = row + key_heads * kd + kh * kd;
        const float *vsrc = row + 2 * key_heads * kd + h * vd;
        float qs = 0.0f, ks = 0.0f;
        for (int i = threadIdx.x; i < kd; i += blockDim.x) {
            qs += qsrc[i] * qsrc[i];
            ks += ksrc[i] * ksrc[i];
        }
        const float qn = sqrtf(block_sum(qs) + eps);
        const float kn = sqrtf(block_sum(ks) + eps);
        const float qscale = 1.0f / (qn * sqrtf(static_cast<float>(kd)));
        for (int i = threadIdx.x; i < kd; i += blockDim.x) {
            qk[i] = qsrc[i] * qscale;
            qk[kd + i] = ksrc[i] / kn;
        }
        __syncthreads();
        const float beta = sigmoid_f(beta_raw[static_cast<size_t>(t) * value_heads + h]);
        const float xa = alpha_raw[static_cast<size_t>(t) * value_heads + h] + dt_bias[h];
        const float softplus = xa > 20.0f ? xa : log1pf(expf(xa));
        const float decay = expf(ssm_a[h] * softplus);
        float *snap_h = (snap != nullptr && t < ntok - 1) ? snap + (static_cast<size_t>(t) * value_heads + h) * kd * vd : nullptr;
        for (int j = threadIdx.x; j < vd; j += blockDim.x) {
            float kv_mem = 0.0f;
            for (int i = 0; i < kd; ++i) {
                const float sij = S[static_cast<size_t>(i) * vd + j] * decay;
                S[static_cast<size_t>(i) * vd + j] = sij;
                kv_mem += sij * qk[kd + i];
            }
            const float delta = (vsrc[j] - kv_mem) * beta;
            float o = 0.0f;
            for (int i = 0; i < kd; ++i) {
                const float sij = S[static_cast<size_t>(i) * vd + j] + qk[kd + i] * delta;
                S[static_cast<size_t>(i) * vd + j] = sij;
                o += sij * qk[i];
                if (snap_h != nullptr) snap_h[static_cast<size_t>(i) * vd + j] = sij;
            }
            out[static_cast<size_t>(t) * value_heads * vd + h * vd + j] = o;
        }
        __syncthreads();  // qk is rewritten for the next token
    }
}

constexpr int kRowsPerBlock = 8;
constexpr int kMaxTokens = 8;
constexpr int kMaxArgmaxRows = 64;

// One warp per row; lanes stride over 8-element chunks so neighbouring lanes read
// neighbouring weight bytes. Each dequantized chunk is applied to NT activation rows
// (x: [NT][cols], y: [NT][rows]), so the weights are read once for all NT tokens.
// Unquantized types are treated as 8-element pseudo-blocks.
template <int TYPE, int NT>
__global__ void vinf_qmatvec_kernel(const uint8_t *w, const float *x, float *y, int rows, int cols, int row_bytes,
                                    int block_size, int type_size) {
    // IQ lookup tables are indexed divergently across lanes; __constant__ memory would
    // serialize those reads, so stage them in shared memory.
    __shared__ int8_t s_iq4nl[16];
    // Codebook of IQ types (IQ3_S, IQ2_*, IQ3_XXS), also read divergently: staged per block.
    __shared__ uint8_t s_iq3s_grid[kGridBytes<TYPE> > 0 ? kGridBytes<TYPE> : 4];
    if constexpr (TYPE == kIQ4_NL || TYPE == kIQ4_XS) {
        if (threadIdx.x < 16) s_iq4nl[threadIdx.x] = c_iq4nl_values[threadIdx.x];
        __syncthreads();
    } else if constexpr (kGridBytes<TYPE> > 0) {
        const uint8_t *src = grid_source<TYPE>();
        for (int i = threadIdx.x; i < kGridBytes<TYPE>; i += blockDim.x) s_iq3s_grid[i] = src[i];
        __syncthreads();
    }
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / 32;
    if (row >= rows) return;
    const uint8_t *w_row = w + static_cast<size_t>(row) * row_bytes;
    const int chunks = cols / 8;
    const int bs = block_size >= 8 ? block_size : 8;
    const int chunk_bytes_unq = 8 * type_size;
    const int chunks_per_block = bs / 8;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    for (int k = lane; k < chunks; k += 32) {
        float v[8];
        if (block_size >= 8) {
            deq8<TYPE>(w_row + static_cast<size_t>(k / chunks_per_block) * type_size, k % chunks_per_block, v, s_iq4nl,
                       s_iq3s_grid);
        } else {
            deq8<TYPE>(w_row + static_cast<size_t>(k) * chunk_bytes_unq, 0, v, s_iq4nl, s_iq3s_grid);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 *xt = reinterpret_cast<const float4 *>(x + static_cast<size_t>(t) * cols);
            const float4 x0 = xt[2 * k];
            const float4 x1 = xt[2 * k + 1];
            acc[t] += v[0] * x0.x + v[1] * x0.y + v[2] * x0.z + v[3] * x0.w + v[4] * x1.x + v[5] * x1.y +
                      v[6] * x1.z + v[7] * x1.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float total = warp_sum(acc[t]);
        if (lane == 0) y[static_cast<size_t>(t) * rows + row] = total;
    }
}

template <int TYPE>
cudaError_t launch_qmatvec_typed(const uint8_t *w, const float *x, float *y, int rows, int cols, int row_bytes,
                                 const TypeTraits &t, int ntok) {
    const int grid = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    const int threads = 32 * kRowsPerBlock;
    switch (ntok) {
        case 1: vinf_qmatvec_kernel<TYPE, 1><<<grid, threads>>>(w, x, y, rows, cols, row_bytes, t.block_size, t.type_size); break;
        case 2: vinf_qmatvec_kernel<TYPE, 2><<<grid, threads>>>(w, x, y, rows, cols, row_bytes, t.block_size, t.type_size); break;
        case 3: vinf_qmatvec_kernel<TYPE, 3><<<grid, threads>>>(w, x, y, rows, cols, row_bytes, t.block_size, t.type_size); break;
        case 4: vinf_qmatvec_kernel<TYPE, 4><<<grid, threads>>>(w, x, y, rows, cols, row_bytes, t.block_size, t.type_size); break;
        case 5: vinf_qmatvec_kernel<TYPE, 5><<<grid, threads>>>(w, x, y, rows, cols, row_bytes, t.block_size, t.type_size); break;
        case 6: vinf_qmatvec_kernel<TYPE, 6><<<grid, threads>>>(w, x, y, rows, cols, row_bytes, t.block_size, t.type_size); break;
        case 7: vinf_qmatvec_kernel<TYPE, 7><<<grid, threads>>>(w, x, y, rows, cols, row_bytes, t.block_size, t.type_size); break;
        case 8: vinf_qmatvec_kernel<TYPE, 8><<<grid, threads>>>(w, x, y, rows, cols, row_bytes, t.block_size, t.type_size); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

// ---- prompt-pass GEMM on tensor cores ------------------------------------------------------------
// Y[t][r] = sum_k W[r][k] X[t][k] for many token rows t. A block owns kGemmBM weight rows x kGemmBN
// tokens: it dequantizes a kGemmBM x kGemmBK weight tile to fp16 in shared memory (each weight read
// once per token tile), takes the fp16 activations straight from global/L2 as the column-major B
// operand, and accumulates 16x16x16 fp16 MMAs in fp32. Every output element is a dot product in the
// same K order whatever the batch size or tiling, so results do not depend on how a prompt is split.
constexpr int kGemmBM = 64, kGemmBN = 64, kGemmBK = 32, kGemmLdA = kGemmBK + 8, kGemmWarps = 4;

__global__ void f32_to_f16_rows_kernel(const float *x, __half *xh, int ntok, int ntok_pad, int cols) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(ntok_pad) * cols) return;
    xh[i] = i < static_cast<size_t>(ntok) * cols ? __float2half_rn(x[i]) : __float2half_rn(0.0f);
}

template <int TYPE>
__global__ void __launch_bounds__(32 * kGemmWarps) vinf_qgemm_kernel(const uint8_t *w, const __half *xh, float *y,
                                                                       int rows, int cols, int row_bytes, int ntok,
                                                                       int block_size, int type_size) {
    using namespace nvcuda;
    __shared__ int8_t s_iq4nl[16];
    __shared__ uint8_t s_iq3s_grid[kGridBytes<TYPE> > 0 ? kGridBytes<TYPE> : 4];
    __shared__ __align__(32) __half a_tile[kGemmBM * kGemmLdA];
    __shared__ __align__(32) float c_tile[kGemmBN * kGemmBM];  // [token][row]
    if constexpr (TYPE == kIQ4_NL || TYPE == kIQ4_XS) {
        if (threadIdx.x < 16) s_iq4nl[threadIdx.x] = c_iq4nl_values[threadIdx.x];
    } else if constexpr (kGridBytes<TYPE> > 0) {
        const uint8_t *src = grid_source<TYPE>();
        for (int i = threadIdx.x; i < kGridBytes<TYPE>; i += blockDim.x) s_iq3s_grid[i] = src[i];
    }
    const int row0 = blockIdx.x * kGemmBM, tok0 = blockIdx.y * kGemmBN;
    const int warp = threadIdx.x / 32;
    const int wm = (warp / 2) * 32, wn = (warp % 2) * 32;  // this warp's 32 x 32 sub-tile
    const int bs = block_size >= 8 ? block_size : 8;
    const int chunks_per_block = bs / 8;
    const int chunk_bytes_unq = 8 * type_size;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    __syncthreads();
    for (int k0 = 0; k0 < cols; k0 += kGemmBK) {
        // Dequantize kGemmBM x kGemmBK weights: 256 chunks of 8 values, 2 per thread.
        for (int c = threadIdx.x; c < kGemmBM * (kGemmBK / 8); c += blockDim.x) {
            const int r = c / (kGemmBK / 8), kc = (k0 / 8) + c % (kGemmBK / 8);
            float v[8];
            if (row0 + r < rows) {
                const uint8_t *w_row = w + static_cast<size_t>(row0 + r) * row_bytes;
                if (block_size >= 8) {
                    deq8<TYPE>(w_row + static_cast<size_t>(kc / chunks_per_block) * type_size, kc % chunks_per_block, v,
                               s_iq4nl, s_iq3s_grid);
                } else {
                    deq8<TYPE>(w_row + static_cast<size_t>(kc) * chunk_bytes_unq, 0, v, s_iq4nl, s_iq3s_grid);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) v[i] = 0.0f;
            }
            __half2 *dst = reinterpret_cast<__half2 *>(a_tile + r * kGemmLdA + (c % (kGemmBK / 8)) * 8);
#pragma unroll
            for (int i = 0; i < 4; ++i) dst[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kGemmBK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], a_tile + (wm + 16 * i) * kGemmLdA + kk, kGemmLdA);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(b[j], xh + static_cast<size_t>(tok0 + wn + 16 * j) * cols + k0 + kk, cols);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    // C[m = row][n = token] -> c_tile[token][row] (column-major in (row, token)), then guarded store.
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(c_tile + (wn + 16 * j) * kGemmBM + wm + 16 * i, acc[i][j], kGemmBM, wmma::mem_col_major);
    __syncthreads();
    for (int i = threadIdx.x; i < kGemmBN * kGemmBM; i += blockDim.x) {
        const int t = i / kGemmBM, r = i % kGemmBM;
        if (tok0 + t < ntok && row0 + r < rows) y[static_cast<size_t>(tok0 + t) * rows + row0 + r] = c_tile[i];
    }
}

template <int TYPE>
cudaError_t launch_qgemm_typed(const uint8_t *w, const __half *xh, float *y, int rows, int cols, int row_bytes,
                               const TypeTraits &t, int ntok) {
    const dim3 grid((rows + kGemmBM - 1) / kGemmBM, (ntok + kGemmBN - 1) / kGemmBN);
    vinf_qgemm_kernel<TYPE><<<grid, 32 * kGemmWarps>>>(w, xh, y, rows, cols, row_bytes, ntok, t.block_size, t.type_size);
    return cudaGetLastError();
}

cudaError_t launch_qgemm(int type, const uint8_t *w, const __half *xh, float *y, int rows, int cols, int row_bytes,
                         const TypeTraits &t, int ntok) {
    switch (type) {
        case kF32: return launch_qgemm_typed<kF32>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kF16: return launch_qgemm_typed<kF16>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kQ8_0: return launch_qgemm_typed<kQ8_0>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kQ2_K: return launch_qgemm_typed<kQ2_K>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kQ3_K: return launch_qgemm_typed<kQ3_K>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kQ4_K: return launch_qgemm_typed<kQ4_K>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kQ5_K: return launch_qgemm_typed<kQ5_K>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kQ6_K: return launch_qgemm_typed<kQ6_K>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kIQ4_NL: return launch_qgemm_typed<kIQ4_NL>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kIQ3_S: return launch_qgemm_typed<kIQ3_S>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kIQ4_XS: return launch_qgemm_typed<kIQ4_XS>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kIQ2_XXS: return launch_qgemm_typed<kIQ2_XXS>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kIQ2_XS: return launch_qgemm_typed<kIQ2_XS>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kIQ2_S: return launch_qgemm_typed<kIQ2_S>(w, xh, y, rows, cols, row_bytes, t, ntok);
        case kIQ3_XXS: return launch_qgemm_typed<kIQ3_XXS>(w, xh, y, rows, cols, row_bytes, t, ntok);
        default: return cudaErrorInvalidValue;
    }
}

cudaError_t launch_qmatvec(int type, const uint8_t *w, const float *x, float *y, int rows, int cols,
                           int row_bytes, const TypeTraits &t, int ntok = 1) {
    switch (type) {
        case kF32: return launch_qmatvec_typed<kF32>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kF16: return launch_qmatvec_typed<kF16>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kQ8_0: return launch_qmatvec_typed<kQ8_0>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kQ2_K: return launch_qmatvec_typed<kQ2_K>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kQ3_K: return launch_qmatvec_typed<kQ3_K>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kQ4_K: return launch_qmatvec_typed<kQ4_K>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kQ5_K: return launch_qmatvec_typed<kQ5_K>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kQ6_K: return launch_qmatvec_typed<kQ6_K>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kIQ4_NL: return launch_qmatvec_typed<kIQ4_NL>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kIQ3_S: return launch_qmatvec_typed<kIQ3_S>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kIQ4_XS: return launch_qmatvec_typed<kIQ4_XS>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kIQ2_XXS: return launch_qmatvec_typed<kIQ2_XXS>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kIQ2_XS: return launch_qmatvec_typed<kIQ2_XS>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kIQ2_S: return launch_qmatvec_typed<kIQ2_S>(w, x, y, rows, cols, row_bytes, t, ntok);
        case kIQ3_XXS: return launch_qmatvec_typed<kIQ3_XXS>(w, x, y, rows, cols, row_bytes, t, ntok);
        default: return cudaErrorInvalidValue;
    }
}

// DFlash 2 grouped dynamic causal convolution over block positions (reference:
// z-lab/dflash GroupedDynamicCausalConv): out[t][c] = sum_{o < K, t >= o}
// (base[o][c] + dyn[t][(sel * K + o) * G + c / group]) * x[t - o][c].
// x, out: [ntok][hidden]; dyn: [ntok][2 * K * G]; base: [K][hidden] (one of the two base kernels).
__global__ void dyn_conv_kernel(const float *x, const float *dyn, const float *base, float *out, int ntok, int hidden,
                                int K, int group, int sel) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntok * hidden) return;
    const int t = i / hidden, c = i % hidden, G = hidden / group;
    float acc = 0.0f;
    for (int o = 0; o < K && o <= t; ++o) {
        const float k = base[static_cast<size_t>(o) * hidden + c] + dyn[static_cast<size_t>(t) * 2 * K * G + (sel * K + o) * G + c / group];
        acc += k * x[static_cast<size_t>(t - o) * hidden + c];
    }
    out[i] = acc;
}

// Row-wise top-k (k <= 32): block t repeatedly extracts the largest remaining value of row t.
__global__ void topk_kernel(const float *x_all, int n, int k, int *out_idx, float *out_val) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    __shared__ int chosen[32];
    const float *x = x_all + static_cast<size_t>(blockIdx.x) * n;
    for (int r = 0; r < k; ++r) {
        float v = -INFINITY;
        int vi = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            bool taken = false;
            for (int j = 0; j < r; ++j) taken |= chosen[j] == i;
            if (!taken && (x[i] > v || (x[i] == v && i < vi))) {
                v = x[i];
                vi = i;
            }
        }
        bv[threadIdx.x] = v;
        bi[threadIdx.x] = vi;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const float ov = bv[threadIdx.x + stride];
                const int oi = bi[threadIdx.x + stride];
                if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                    bv[threadIdx.x] = ov;
                    bi[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            chosen[r] = bi[0];
            out_idx[blockIdx.x * k + r] = bi[0];
            out_val[blockIdx.x * k + r] = bv[0];
        }
        __syncthreads();
    }
}

// Row-wise argmax: block t reduces row t of x ([ntok][n]); ties resolve to the lowest index.
__global__ void argmax_kernel(const float *x_all, int n, int *out_all) {
    const float *x = x_all + static_cast<size_t>(blockIdx.x) * n;
    int *out = out_all + blockIdx.x;
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > bv) { bv = x[i]; bi = i; }
    }
    best_v[threadIdx.x] = bv;
    best_i[threadIdx.x] = bi;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float ov = best_v[threadIdx.x + stride];
            const int oi = best_i[threadIdx.x + stride];
            if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
                best_v[threadIdx.x] = ov;
                best_i[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = best_i[0];
}

struct DeviceTensor {
    uint8_t *data = nullptr;  // device pointer when resident, pinned host pointer when streamed
    bool resident = true;
    bool in_arena = false;  // carved from the weight arena (freed with it)
    int type = 0;
    int rows = 0;
    int cols = 0;
    int row_bytes = 0;
    size_t nbytes = 0;
};

struct DeviceBuffer {
    float *data = nullptr;
    size_t n = 0;     // elements
    int elem = 4;     // bytes per element: 4 (fp32) or 2 (fp16 KV caches)
    size_t bytes() const { return n * static_cast<size_t>(elem); }
};

// Prefetch ring for streamed tensors. Streamed weights are used in a fixed cyclic order
// (the decode op sequence), so item s of the sequence lives in slot s % K. The copy stream
// (non-blocking) fills a slot once the kernel that last read it has finished; each matvec
// waits only for its own slot's copy. Out-of-order use resynchronizes the ring.
struct StreamRing {
    std::vector<std::string> order;
    std::unordered_map<std::string, int> index;
    std::vector<uint8_t *> slots;
    std::vector<cudaEvent_t> ready;
    std::vector<cudaEvent_t> free_ev;
    size_t slot_bytes = 0;
    cudaStream_t copy = nullptr;
    long long use_seq = 0;
    long long issue_seq = 0;
    int base = 0;
    bool primed = false;

    int K() const { return static_cast<int>(slots.size()); }
    int N() const { return static_cast<int>(order.size()); }

    void destroy() {
        if (copy != nullptr) cudaStreamSynchronize(copy);
        for (auto *p : slots) cudaFree(p);
        for (auto e : ready) cudaEventDestroy(e);
        for (auto e : free_ev) cudaEventDestroy(e);
        if (copy != nullptr) cudaStreamDestroy(copy);
        slots.clear();
        ready.clear();
        free_ev.clear();
        copy = nullptr;
    }
};

struct RuntimeObject {
    PyObject_HEAD
    StreamRing *ring;
    std::unordered_map<std::string, DeviceTensor> *tensors;
    std::unordered_map<std::string, DeviceBuffer> *buffers;
    uint8_t *staging;
    size_t staging_bytes;
    uint8_t *arena;  // one allocation for all resident weights (no per-tensor allocator rounding)
    size_t arena_bytes;
    size_t arena_used;
    int *argmax_out;
    unsigned long long streamed_bytes_total;
    float *scratch_x;
    float *scratch_y;
    size_t scratch_x_len;
    size_t scratch_y_len;
    __half *gemm_x;  // fp16 activations for the tensor-core GEMM, [ntok rounded to kGemmBN][cols]
    size_t gemm_x_len;
    int gemm_min_rows;  // prompt-pass kernels (GEMM, tiled attention) from this many rows (0 = more than kMaxTokens)
    // Multi-sequence passes: the rows of a pass are segments of consecutive positions of one sequence slot.
    std::vector<int> *seg;  // [slot, pos0, rows, row0] per segment
    const int *page_table;  // device [slots][pages_per_seq] page ids (int32 stored in a float buffer)
    int page_size;
    int pages_per_seq;
    int max_slots;
};

PyObject *cuda_error(const char *context, cudaError_t err) {
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s", context, cudaGetErrorString(err));
    return nullptr;
}

bool ensure_scratch(RuntimeObject *self, size_t x_len, size_t y_len) {
    if (x_len > self->scratch_x_len) {
        cudaFree(self->scratch_x);
        self->scratch_x = nullptr;
        if (cudaMalloc(&self->scratch_x, x_len * sizeof(float)) != cudaSuccess) return false;
        self->scratch_x_len = x_len;
    }
    if (y_len > self->scratch_y_len) {
        cudaFree(self->scratch_y);
        self->scratch_y = nullptr;
        if (cudaMalloc(&self->scratch_y, y_len * sizeof(float)) != cudaSuccess) return false;
        self->scratch_y_len = y_len;
    }
    return true;
}

void free_tensor(DeviceTensor &t) {
    if (t.in_arena) {
        // Arena space is reclaimed only when the arena is reset.
    } else if (t.resident) {
        cudaFree(t.data);
    } else {
        cudaFreeHost(t.data);
    }
    t.data = nullptr;
}

// Device pointer for a tensor's blocks; streamed tensors are copied into the staging buffer first.
cudaError_t tensor_device_ptr(RuntimeObject *self, DeviceTensor &t, const uint8_t **out) {
    if (t.resident) {
        *out = t.data;
        return cudaSuccess;
    }
    if (t.nbytes > self->staging_bytes) return cudaErrorInvalidValue;
    cudaError_t err = cudaMemcpyAsync(self->staging, t.data, t.nbytes, cudaMemcpyHostToDevice, 0);
    self->streamed_bytes_total += t.nbytes;
    *out = self->staging;
    return err;
}

cudaError_t ring_issue(RuntimeObject *self) {
    StreamRing &r = *self->ring;
    const int slot = static_cast<int>(r.issue_seq % r.K());
    const std::string &name = r.order[(r.base + r.issue_seq) % r.N()];
    DeviceTensor &t = (*self->tensors)[name];
    cudaError_t err = cudaStreamWaitEvent(r.copy, r.free_ev[slot], 0);
    if (err == cudaSuccess) err = cudaMemcpyAsync(r.slots[slot], t.data, t.nbytes, cudaMemcpyHostToDevice, r.copy);
    if (err == cudaSuccess) err = cudaEventRecord(r.ready[slot], r.copy);
    self->streamed_bytes_total += t.nbytes;
    ++r.issue_seq;
    return err;
}

cudaError_t ring_resync(RuntimeObject *self, int order_index) {
    StreamRing &r = *self->ring;
    cudaError_t err = cudaDeviceSynchronize();
    r.base = order_index;
    r.use_seq = 0;
    r.issue_seq = 0;
    r.primed = true;
    for (int i = 0; i < r.K() && err == cudaSuccess; ++i) err = ring_issue(self);
    return err;
}

// Ring path for a streamed tensor; *slot_out receives the slot whose free event must be
// recorded after the consuming kernel.
cudaError_t ring_acquire(RuntimeObject *self, const std::string &name, const uint8_t **out, int *slot_out) {
    StreamRing &r = *self->ring;
    const int j = r.index[name];
    cudaError_t err = cudaSuccess;
    if (!r.primed || (r.base + r.use_seq) % r.N() != j) err = ring_resync(self, j);
    if (err != cudaSuccess) return err;
    const int slot = static_cast<int>(r.use_seq % r.K());
    err = cudaStreamWaitEvent(0, r.ready[slot], 0);
    *out = r.slots[slot];
    *slot_out = slot;
    return err;
}

cudaError_t ring_release(RuntimeObject *self, int slot) {
    StreamRing &r = *self->ring;
    cudaError_t err = cudaEventRecord(r.free_ev[slot], 0);
    ++r.use_seq;
    while (err == cudaSuccess && r.issue_seq < r.use_seq + r.K()) err = ring_issue(self);
    return err;
}

bool ring_has(RuntimeObject *self, const char *name) {
    return self->ring != nullptr && self->ring->K() > 0 && self->ring->index.count(name) != 0;
}

DeviceTensor *find_tensor(RuntimeObject *self, const char *name) {
    auto found = self->tensors->find(name);
    if (found == self->tensors->end()) {
        PyErr_Format(PyExc_KeyError, "weight tensor %s is not loaded", name);
        return nullptr;
    }
    return &found->second;
}

float *find_buffer(RuntimeObject *self, const char *name, size_t min_n) {
    auto found = self->buffers->find(name);
    if (found == self->buffers->end()) {
        PyErr_Format(PyExc_KeyError, "device buffer %s is not allocated", name);
        return nullptr;
    }
    if (found->second.elem != 4) {
        PyErr_Format(PyExc_TypeError, "device buffer %s is fp16 (KV cache), not fp32", name);
        return nullptr;
    }
    if (found->second.n < min_n) {
        PyErr_Format(PyExc_ValueError, "device buffer %s has %zu floats, need %zu", name, found->second.n, min_n);
        return nullptr;
    }
    return found->second.data;
}

// KV cache buffers may be fp32 or fp16; returns the raw pointer and the element size.
void *find_kv_buffer(RuntimeObject *self, const char *name, size_t min_n, int *elem) {
    auto found = self->buffers->find(name);
    if (found == self->buffers->end()) {
        PyErr_Format(PyExc_KeyError, "device buffer %s is not allocated", name);
        return nullptr;
    }
    if (found->second.n < min_n) {
        PyErr_Format(PyExc_ValueError, "KV buffer %s has %zu elements, need %zu", name, found->second.n, min_n);
        return nullptr;
    }
    *elem = found->second.elem;
    return found->second.data;
}

DeviceBuffer *find_any_buffer(RuntimeObject *self, const char *name) {
    auto found = self->buffers->find(name);
    if (found == self->buffers->end()) {
        PyErr_Format(PyExc_KeyError, "device buffer %s is not allocated", name);
        return nullptr;
    }
    return &found->second;
}

// F32 weight tensors used directly by elementwise kernels (norm weights, ssm_a, conv1d, ...).
const float *find_f32_weight(RuntimeObject *self, const char *name, size_t min_n) {
    DeviceTensor *t = find_tensor(self, name);
    if (t == nullptr) return nullptr;
    if (t->type != kF32 || !t->resident) {
        PyErr_Format(PyExc_ValueError, "%s must be a GPU-resident F32 tensor", name);
        return nullptr;
    }
    if (static_cast<size_t>(t->rows) * t->cols < min_n) {
        PyErr_Format(PyExc_ValueError, "%s has %d elements, need %zu", name, t->rows * t->cols, min_n);
        return nullptr;
    }
    return reinterpret_cast<const float *>(t->data);
}

PyObject *launch_result(const char *context) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return cuda_error(context, err);
    Py_RETURN_NONE;
}

int blocks_for(size_t n, int threads) { return static_cast<int>((n + threads - 1) / threads); }

int Runtime_init(RuntimeObject *self, PyObject *args, PyObject *kwargs) {
    static const char *kwlist[] = {"iq3s_grid", nullptr};
    Py_buffer grid;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*", const_cast<char **>(kwlist), &grid)) return -1;
    if (grid.len != 512 * 4) {
        PyBuffer_Release(&grid);
        PyErr_SetString(PyExc_ValueError, "iq3s_grid must be 2048 bytes (512 entries x 4 values)");
        return -1;
    }
    cudaError_t err = cudaMemcpyToSymbol(c_iq3s_grid, grid.buf, 512 * 4);
    PyBuffer_Release(&grid);
    if (err != cudaSuccess) {
        cuda_error("cudaMemcpyToSymbol iq3s grid", err);
        return -1;
    }
    if (self->tensors == nullptr) self->tensors = new std::unordered_map<std::string, DeviceTensor>();
    if (self->buffers == nullptr) self->buffers = new std::unordered_map<std::string, DeviceBuffer>();
    if (self->argmax_out == nullptr) {
        err = cudaMalloc(&self->argmax_out, kMaxArgmaxRows * sizeof(int));
        if (err != cudaSuccess) {
            cuda_error("cudaMalloc argmax", err);
            return -1;
        }
    }
    return 0;
}

void Runtime_dealloc(RuntimeObject *self) {
    if (self->tensors != nullptr) {
        for (auto &item : *self->tensors) free_tensor(item.second);
        delete self->tensors;
        self->tensors = nullptr;
    }
    if (self->buffers != nullptr) {
        for (auto &item : *self->buffers) cudaFree(item.second.data);
        delete self->buffers;
        self->buffers = nullptr;
    }
    cudaFree(self->staging);
    cudaFree(self->arena);
    cudaFree(self->argmax_out);
    if (self->ring != nullptr) {
        self->ring->destroy();
        delete self->ring;
        self->ring = nullptr;
    }
    cudaFree(self->scratch_x);
    cudaFree(self->scratch_y);
    cudaFree(self->gemm_x);
    delete self->seg;
    self->seg = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *Runtime_upload(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    Py_buffer data;
    int type = 0, cols = 0, rows = 0, resident = 1;
    if (!PyArg_ParseTuple(args, "sy*iii|p", &name, &data, &type, &cols, &rows, &resident)) return nullptr;
    TypeTraits t;
    if (!type_traits(type, &t)) {
        PyBuffer_Release(&data);
        return PyErr_Format(PyExc_ValueError, "unsupported GGUF tensor type id %d for CUDA matvec", type);
    }
    if (cols <= 0 || rows <= 0 || cols % t.block_size != 0) {
        PyBuffer_Release(&data);
        return PyErr_Format(PyExc_ValueError, "%s: cols=%d must be a positive multiple of the block size", name, cols);
    }
    const size_t row_bytes = static_cast<size_t>(cols / t.block_size) * t.type_size;
    const size_t nbytes = row_bytes * static_cast<size_t>(rows);
    if (static_cast<size_t>(data.len) != nbytes) {
        PyBuffer_Release(&data);
        return PyErr_Format(PyExc_ValueError, "%s: got %zd bytes, expected %zu", name, data.len, nbytes);
    }
    auto found = self->tensors->find(name);
    if (found != self->tensors->end()) {
        free_tensor(found->second);
        self->tensors->erase(found);
    }
    DeviceTensor dt;
    dt.resident = resident != 0;
    cudaError_t err;
    // +16: the megakernel's aligned 16-byte row loads may read up to 15 bytes past the end.
    const size_t padded = (nbytes + 16 + 255) / 256 * 256;
    if (dt.resident && self->arena != nullptr && self->arena_used + padded <= self->arena_bytes) {
        dt.data = self->arena + self->arena_used;
        dt.in_arena = true;
        self->arena_used += padded;
        err = cudaMemcpy(dt.data, data.buf, nbytes, cudaMemcpyHostToDevice);
    } else if (dt.resident) {
        err = cudaMalloc(&dt.data, nbytes + 16);
        if (err == cudaSuccess) err = cudaMemcpy(dt.data, data.buf, nbytes, cudaMemcpyHostToDevice);
    } else {
        // Mapped so the fused megakernel can read it directly (UVA: host ptr == device ptr).
        err = cudaHostAlloc(reinterpret_cast<void **>(&dt.data), nbytes, cudaHostAllocMapped | cudaHostAllocPortable);
        if (err == cudaSuccess) memcpy(dt.data, data.buf, nbytes);
    }
    PyBuffer_Release(&data);
    if (err != cudaSuccess) {
        free_tensor(dt);
        return cuda_error("upload tensor", err);
    }
    dt.type = type;
    dt.rows = rows;
    dt.cols = cols;
    dt.row_bytes = static_cast<int>(row_bytes);
    dt.nbytes = nbytes;
    (*self->tensors)[name] = dt;
    Py_RETURN_NONE;
}

PyObject *Runtime_free(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    if (!PyArg_ParseTuple(args, "s", &name)) return nullptr;
    auto found = self->tensors->find(name);
    if (found == self->tensors->end()) return PyErr_Format(PyExc_KeyError, "%s", name);
    if (ring_has(self, name)) return PyErr_Format(PyExc_ValueError, "%s is in the stream order; reset it first", name);
    free_tensor(found->second);
    self->tensors->erase(found);
    Py_RETURN_NONE;
}

PyObject *Runtime_has(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    if (!PyArg_ParseTuple(args, "s", &name)) return nullptr;
    return PyBool_FromLong(self->tensors->count(name) != 0);
}

PyObject *Runtime_device_bytes(RuntimeObject *self, PyObject *) {
    size_t total = 0;
    for (auto &item : *self->tensors) if (item.second.resident) total += item.second.nbytes;
    return PyLong_FromSize_t(total);
}

PyObject *Runtime_pinned_bytes(RuntimeObject *self, PyObject *) {
    size_t total = 0;
    for (auto &item : *self->tensors) if (!item.second.resident) total += item.second.nbytes;
    return PyLong_FromSize_t(total);
}

PyObject *Runtime_streamed_bytes(RuntimeObject *self, PyObject *) {
    return PyLong_FromUnsignedLongLong(self->streamed_bytes_total);
}

// tensor_info(name) -> (device_address, gguf_type, rows, cols, row_bytes, nbytes, resident)
// Streamed tensors report their host-mapped address (readable from kernels via UVA).
PyObject *Runtime_tensor_info(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    if (!PyArg_ParseTuple(args, "s", &name)) return nullptr;
    DeviceTensor *t = find_tensor(self, name);
    if (t == nullptr) return nullptr;
    void *addr = t->data;
    if (!t->resident) {
        cudaError_t err = cudaHostGetDevicePointer(&addr, t->data, 0);
        if (err != cudaSuccess) return cuda_error("cudaHostGetDevicePointer", err);
    }
    return Py_BuildValue("(KiiiinO)", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(addr)), t->type,
                         t->rows, t->cols, t->row_bytes, static_cast<Py_ssize_t>(t->nbytes),
                         t->resident ? Py_True : Py_False);
}

// buffer_ptr(name) -> (device_address, n_floats)
PyObject *Runtime_buffer_ptr(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    if (!PyArg_ParseTuple(args, "s", &name)) return nullptr;
    float *buf = find_buffer(self, name, 0);
    if (buf == nullptr) return nullptr;
    return Py_BuildValue("(Kn)", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(buf)),
                         static_cast<Py_ssize_t>((*self->buffers)[name].n));
}

// set_arena(nbytes): one device allocation that resident uploads are carved from (256-byte aligned).
PyObject *Runtime_set_arena(RuntimeObject *self, PyObject *args) {
    Py_ssize_t nbytes = 0;
    if (!PyArg_ParseTuple(args, "n", &nbytes)) return nullptr;
    for (auto &item : *self->tensors)
        if (item.second.in_arena) return PyErr_Format(PyExc_RuntimeError, "arena is in use by %s", item.first.c_str());
    cudaFree(self->arena);
    self->arena = nullptr;
    self->arena_bytes = self->arena_used = 0;
    if (nbytes > 0) {
        cudaError_t err = cudaMalloc(&self->arena, nbytes);
        if (err != cudaSuccess) return cuda_error("cudaMalloc weight arena", err);
        self->arena_bytes = static_cast<size_t>(nbytes);
    }
    Py_RETURN_NONE;
}

PyObject *Runtime_staging_bytes(RuntimeObject *self, PyObject *) {
    size_t total = self->staging_bytes;
    if (self->ring != nullptr) total += self->ring->slot_bytes * self->ring->slots.size();
    return PyLong_FromSize_t(total);
}

PyObject *Runtime_set_staging(RuntimeObject *self, PyObject *args) {
    Py_ssize_t nbytes = 0;
    if (!PyArg_ParseTuple(args, "n", &nbytes)) return nullptr;
    cudaFree(self->staging);
    self->staging = nullptr;
    self->staging_bytes = 0;
    if (nbytes > 0) {
        cudaError_t err = cudaMalloc(&self->staging, nbytes);
        if (err != cudaSuccess) return cuda_error("cudaMalloc staging", err);
        self->staging_bytes = static_cast<size_t>(nbytes);
    }
    Py_RETURN_NONE;
}

// set_stream_order(names, slots): enable prefetching for streamed tensors used in this cyclic order.
PyObject *Runtime_set_stream_order(RuntimeObject *self, PyObject *args) {
    PyObject *names = nullptr;
    int slots = 0;
    if (!PyArg_ParseTuple(args, "Oi", &names, &slots)) return nullptr;
    PyObject *fast = PySequence_Fast(names, "names must be a sequence of str");
    if (fast == nullptr) return nullptr;
    cudaDeviceSynchronize();
    if (self->ring != nullptr) {
        self->ring->destroy();
        delete self->ring;
    }
    self->ring = new StreamRing();
    StreamRing &r = *self->ring;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    for (Py_ssize_t i = 0; i < n; ++i) {
        const char *name = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(fast, i));
        if (name == nullptr) {
            Py_DECREF(fast);
            return nullptr;
        }
        auto found = self->tensors->find(name);
        if (found == self->tensors->end() || found->second.resident) {
            Py_DECREF(fast);
            return PyErr_Format(PyExc_ValueError, "%s is not a loaded streamed tensor", name);
        }
        if (r.index.count(name) != 0) {
            Py_DECREF(fast);
            return PyErr_Format(PyExc_ValueError, "%s appears twice in the stream order", name);
        }
        r.index[name] = static_cast<int>(r.order.size());
        r.order.push_back(name);
        if (found->second.nbytes > r.slot_bytes) r.slot_bytes = found->second.nbytes;
    }
    Py_DECREF(fast);
    if (n == 0 || slots <= 0) Py_RETURN_NONE;
    if (slots > n) slots = static_cast<int>(n);
    cudaError_t err = cudaStreamCreateWithFlags(&r.copy, cudaStreamNonBlocking);
    for (int i = 0; i < slots && err == cudaSuccess; ++i) {
        uint8_t *p = nullptr;
        cudaEvent_t a = nullptr, b = nullptr;
        err = cudaMalloc(&p, r.slot_bytes);
        if (err == cudaSuccess) err = cudaEventCreateWithFlags(&a, cudaEventDisableTiming);
        if (err == cudaSuccess) err = cudaEventCreateWithFlags(&b, cudaEventDisableTiming);
        r.slots.push_back(p);
        r.ready.push_back(a);
        r.free_ev.push_back(b);
    }
    if (err != cudaSuccess) {
        r.destroy();
        return cuda_error("stream ring allocation", err);
    }
    Py_RETURN_NONE;
}

PyObject *Runtime_alloc(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    Py_ssize_t n = 0;
    int elem = 4;
    if (!PyArg_ParseTuple(args, "sn|i", &name, &n, &elem)) return nullptr;
    if (n <= 0) return PyErr_Format(PyExc_ValueError, "buffer %s size must be positive", name);
    if (elem != 4 && elem != 2) return PyErr_Format(PyExc_ValueError, "element size must be 4 (fp32) or 2 (fp16)");
    auto found = self->buffers->find(name);
    if (found != self->buffers->end()) {
        cudaFree(found->second.data);
        self->buffers->erase(found);
    }
    DeviceBuffer b;
    b.elem = elem;
    cudaError_t err = cudaMalloc(&b.data, static_cast<size_t>(n) * elem);
    if (err == cudaSuccess) err = cudaMemset(b.data, 0, static_cast<size_t>(n) * elem);
    if (err != cudaSuccess) {
        cudaFree(b.data);
        return cuda_error("cudaMalloc buffer", err);
    }
    b.n = static_cast<size_t>(n);
    (*self->buffers)[name] = b;
    Py_RETURN_NONE;
}

PyObject *Runtime_zero(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    if (!PyArg_ParseTuple(args, "s", &name)) return nullptr;
    DeviceBuffer *b = find_any_buffer(self, name);
    if (b == nullptr) return nullptr;
    cudaError_t err = cudaMemset(b->data, 0, b->bytes());
    if (err != cudaSuccess) return cuda_error("cudaMemset", err);
    Py_RETURN_NONE;
}

PyObject *Runtime_buffer_bytes(RuntimeObject *self, PyObject *) {
    size_t total = 0;
    for (auto &item : *self->buffers) total += item.second.bytes();
    return PyLong_FromSize_t(total);
}

PyObject *Runtime_write(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    Py_buffer data;
    Py_ssize_t offset = 0;
    if (!PyArg_ParseTuple(args, "sy*|n", &name, &data, &offset)) return nullptr;
    // Offsets and sizes are in buffer elements (float32, or float16 for fp16 KV caches).
    DeviceBuffer *b = find_any_buffer(self, name);
    if (b == nullptr || offset < 0 || data.len % b->elem != 0 ||
        static_cast<size_t>(offset) + static_cast<size_t>(data.len) / b->elem > b->n) {
        PyBuffer_Release(&data);
        if (b != nullptr) PyErr_Format(PyExc_ValueError, "write to %s: bad offset/size for %d-byte elements", name, b->elem);
        return nullptr;
    }
    cudaError_t err = cudaMemcpy(reinterpret_cast<char *>(b->data) + static_cast<size_t>(offset) * b->elem, data.buf,
                                 data.len, cudaMemcpyHostToDevice);
    PyBuffer_Release(&data);
    if (err != cudaSuccess) return cuda_error("buffer write", err);
    Py_RETURN_NONE;
}

PyObject *Runtime_read(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    Py_ssize_t n = -1, offset = 0;
    if (!PyArg_ParseTuple(args, "s|nn", &name, &n, &offset)) return nullptr;
    DeviceBuffer *b = find_any_buffer(self, name);
    if (b == nullptr) return nullptr;
    const size_t avail = b->n;
    if (offset < 0 || static_cast<size_t>(offset) > avail) return PyErr_Format(PyExc_ValueError, "bad read offset");
    const size_t count = n < 0 ? avail - offset : static_cast<size_t>(n);
    if (offset + count > avail) return PyErr_Format(PyExc_ValueError, "read of %zu elements exceeds buffer %s", count, name);
    PyObject *out = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(count * b->elem));
    if (out == nullptr) return nullptr;
    cudaError_t err = cudaMemcpy(PyBytes_AS_STRING(out), reinterpret_cast<char *>(b->data) + static_cast<size_t>(offset) * b->elem,
                                 count * b->elem, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        Py_DECREF(out);
        return cuda_error("buffer read", err);
    }
    return out;
}

// qmv(weight, x_buf, y_buf): y = W x on device buffers (streams the weight if not resident).
PyObject *Runtime_qmv(RuntimeObject *self, PyObject *args) {
    const char *wname = nullptr, *xname = nullptr, *yname = nullptr;
    int ntok = 1;
    if (!PyArg_ParseTuple(args, "sss|i", &wname, &xname, &yname, &ntok)) return nullptr;
    if (ntok < 1) return PyErr_Format(PyExc_ValueError, "ntok must be positive");
    DeviceTensor *t = find_tensor(self, wname);
    if (t == nullptr) return nullptr;
    if (t->cols % 32 != 0) return PyErr_Format(PyExc_ValueError, "%s: matvec cols must be a multiple of 32", wname);
    const float *x = find_buffer(self, xname, static_cast<size_t>(t->cols) * ntok);
    if (x == nullptr) return nullptr;
    float *y = find_buffer(self, yname, static_cast<size_t>(t->rows) * ntok);
    if (y == nullptr) return nullptr;
    const uint8_t *w = nullptr;
    int slot = -1;
    cudaError_t err;
    if (!t->resident && ring_has(self, wname)) {
        err = ring_acquire(self, wname, &w, &slot);
    } else {
        err = tensor_device_ptr(self, *t, &w);
    }
    if (err != cudaSuccess) return cuda_error(t->resident ? "qmv" : "qmv stream (staging too small?)", err);
    TypeTraits tt;
    type_traits(t->type, &tt);
    // The GEMM only runs in prompt mode (gemm_min_rows > 0): decode and verification rows keep the matvec
    // kernels, so a sequence's tokens do not depend on how many rows share its pass.
    if (self->gemm_min_rows > 0 && ntok >= self->gemm_min_rows) {
        // Prompt passes: tensor-core GEMM; the (possibly streamed) weight crosses PCIe once for all rows.
        const int ntok_pad = (ntok + kGemmBN - 1) / kGemmBN * kGemmBN;
        const size_t need = static_cast<size_t>(ntok_pad) * t->cols;
        if (need > self->gemm_x_len) {
            cudaFree(self->gemm_x);
            self->gemm_x = nullptr;
            self->gemm_x_len = 0;
            err = cudaMalloc(&self->gemm_x, need * sizeof(__half));
            if (err == cudaSuccess) self->gemm_x_len = need;
        }
        if (err == cudaSuccess) {
            f32_to_f16_rows_kernel<<<blocks_for(need, 256), 256>>>(x, self->gemm_x, ntok, ntok_pad, t->cols);
            err = launch_qgemm(t->type, w, self->gemm_x, y, t->rows, t->cols, t->row_bytes, tt, ntok);
        }
        ntok = 0;  // done
    }
    // Wide passes on the matvec kernel: every group of kMaxTokens rows reuses the weight already in VRAM.
    for (int c = 0; c < ntok && err == cudaSuccess; c += kMaxTokens) {
        const int m = ntok - c < kMaxTokens ? ntok - c : kMaxTokens;
        err = launch_qmatvec(t->type, w, x + static_cast<size_t>(c) * t->cols, y + static_cast<size_t>(c) * t->rows,
                             t->rows, t->cols, t->row_bytes, tt, m);
    }
    if (err == cudaSuccess && slot >= 0) err = ring_release(self, slot);
    if (err != cudaSuccess) return cuda_error("qmv launch", err);
    Py_RETURN_NONE;
}

PyObject *Runtime_rmsnorm(RuntimeObject *self, PyObject *args) {
    const char *xname, *wname, *yname;
    int n = 0, groups = 1;
    float eps = 1e-6f;
    if (!PyArg_ParseTuple(args, "sssif|i", &xname, &wname, &yname, &n, &eps, &groups)) return nullptr;
    const float *x = find_buffer(self, xname, static_cast<size_t>(n) * groups);
    float *y = x ? find_buffer(self, yname, static_cast<size_t>(n) * groups) : nullptr;
    const float *w = y ? find_f32_weight(self, wname, n) : nullptr;
    if (w == nullptr) return nullptr;
    rmsnorm_kernel<<<groups, 256>>>(x, w, y, n, eps);
    return launch_result("rmsnorm");
}

PyObject *Runtime_gated_rmsnorm(RuntimeObject *self, PyObject *args) {
    const char *xname, *wname, *zname, *yname;
    int n = 0, groups = 1;
    float eps = 1e-6f;
    if (!PyArg_ParseTuple(args, "ssssifi", &xname, &wname, &zname, &yname, &n, &eps, &groups)) return nullptr;
    const size_t total = static_cast<size_t>(n) * groups;
    const float *x = find_buffer(self, xname, total);
    const float *z = x ? find_buffer(self, zname, total) : nullptr;
    float *y = z ? find_buffer(self, yname, total) : nullptr;
    const float *w = y ? find_f32_weight(self, wname, n) : nullptr;
    if (w == nullptr) return nullptr;
    gated_rmsnorm_kernel<<<groups, 128>>>(x, w, z, y, n, eps);
    return launch_result("gated_rmsnorm");
}

PyObject *binary_op(RuntimeObject *self, PyObject *args, int op) {
    const char *aname, *bname, *oname;
    int n = 0;
    if (!PyArg_ParseTuple(args, "sssi", &aname, &bname, &oname, &n)) return nullptr;
    const float *a = find_buffer(self, aname, n);
    const float *b = a ? find_buffer(self, bname, n) : nullptr;
    float *o = b ? find_buffer(self, oname, n) : nullptr;
    if (o == nullptr) return nullptr;
    if (op == 0) add_kernel<<<blocks_for(n, 256), 256>>>(a, b, o, n);
    else if (op == 1) silu_mul_kernel<<<blocks_for(n, 256), 256>>>(a, b, o, n);
    else sigmoid_mul_kernel<<<blocks_for(n, 256), 256>>>(a, b, o, n);
    return launch_result("elementwise");
}

PyObject *Runtime_add(RuntimeObject *self, PyObject *args) { return binary_op(self, args, 0); }
PyObject *Runtime_silu_mul(RuntimeObject *self, PyObject *args) { return binary_op(self, args, 1); }
PyObject *Runtime_sigmoid_mul(RuntimeObject *self, PyObject *args) { return binary_op(self, args, 2); }

PyObject *Runtime_split_gated_q(RuntimeObject *self, PyObject *args) {
    const char *rname, *qname, *gname;
    int heads = 0, hd = 0;
    if (!PyArg_ParseTuple(args, "sssii", &rname, &qname, &gname, &heads, &hd)) return nullptr;
    const size_t n = static_cast<size_t>(heads) * hd;
    const float *raw = find_buffer(self, rname, 2 * n);
    float *q = raw ? find_buffer(self, qname, n) : nullptr;
    float *g = q ? find_buffer(self, gname, n) : nullptr;
    if (g == nullptr) return nullptr;
    split_gated_q_kernel<<<blocks_for(n, 256), 256>>>(raw, q, g, heads, hd);
    return launch_result("split_gated_q");
}

PyObject *Runtime_rope(RuntimeObject *self, PyObject *args) {
    const char *xname;
    int heads = 0, hd = 0, rot = 0, pos = 0, ntok = 1;
    double base = 10000.0;
    if (!PyArg_ParseTuple(args, "siiiid|i", &xname, &heads, &hd, &rot, &pos, &base, &ntok)) return nullptr;
    if (rot <= 0 || rot % 2 != 0 || rot > hd) return PyErr_Format(PyExc_ValueError, "invalid rotary dim %d", rot);
    if (ntok < 1) return PyErr_Format(PyExc_ValueError, "ntok must be positive");
    float *x = find_buffer(self, xname, static_cast<size_t>(heads) * hd * ntok);
    if (x == nullptr) return nullptr;
    rope_neox_kernel<<<blocks_for(static_cast<size_t>(ntok) * heads * rot / 2, 128), 128>>>(x, ntok, heads, hd, rot, pos, base);
    return launch_result("rope");
}

PyObject *Runtime_kv_append(RuntimeObject *self, PyObject *args) {
    const char *kname, *vname, *kcname, *vcname;
    int kv_heads = 0, max_seq = 0, hd = 0, pos = 0, ntok = 1;
    if (!PyArg_ParseTuple(args, "ssssiiii|i", &kname, &vname, &kcname, &vcname, &kv_heads, &max_seq, &hd, &pos, &ntok))
        return nullptr;
    if (ntok < 1 || pos < 0 || pos + ntok > max_seq)
        return PyErr_Format(PyExc_ValueError, "positions [%d, %d) outside KV cache of %d", pos, pos + ntok, max_seq);
    const size_t cache = static_cast<size_t>(kv_heads) * hd * max_seq;
    const size_t n = static_cast<size_t>(kv_heads) * hd * ntok;
    int ke = 0, ve = 0;
    const float *k = find_buffer(self, kname, n);
    const float *v = k ? find_buffer(self, vname, n) : nullptr;
    void *kc = v ? find_kv_buffer(self, kcname, cache, &ke) : nullptr;
    void *vc = kc ? find_kv_buffer(self, vcname, cache, &ve) : nullptr;
    if (vc == nullptr) return nullptr;
    if (ke != ve) return PyErr_Format(PyExc_TypeError, "K and V caches differ in precision");
    if (ke == 2)
        kv_append_kernel<__half><<<blocks_for(n, 256), 256>>>(k, v, static_cast<__half *>(kc), static_cast<__half *>(vc),
                                                              ntok, kv_heads, contiguous_kv(max_seq), hd, pos);
    else
        kv_append_kernel<float><<<blocks_for(n, 256), 256>>>(k, v, static_cast<float *>(kc), static_cast<float *>(vc),
                                                             ntok, kv_heads, contiguous_kv(max_seq), hd, pos);
    return launch_result("kv_append");
}

template <typename T>
PyObject *launch_attention(const T *kc, const T *vc, const float *q, float *o, int heads, int kv_heads, int hd,
                           KvAddr A, int seq_len, int ntok, int bidirectional, int window, bool prompt_pass) {
    const size_t smem = static_cast<size_t>(seq_len + ntok - 1) * sizeof(float);
    const float scale = 1.0f / sqrtf(static_cast<float>(hd));
    if ((ntok > kMaxTokens || prompt_pass) && hd % 16 == 0 && attn_tc_smem(hd) <= 99 * 1024) {
        const size_t csmem = attn_tc_smem(hd);
        static size_t tc_attr = 0;
        if (csmem > 48 * 1024 && csmem > tc_attr) {
            cudaError_t err = cudaFuncSetAttribute(attention_tc_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   static_cast<int>(csmem));
            if (err != cudaSuccess) return cuda_error("attention smem attribute", err);
            tc_attr = csmem;
        }
        attention_tc_kernel<T><<<dim3(heads, (ntok + kTcQ - 1) / kTcQ), 32 * kTcWarps, csmem>>>(
            q, kc, vc, o, heads, kv_heads, hd, A, seq_len, ntok, scale, bidirectional, window);
        return launch_result("attention");
    }
    if (ntok > kMaxTokens || smem > 32 * 1024 || prompt_pass) {
        const size_t tsmem = attn_tiled_smem(hd);
        if (hd > 1024 || tsmem > 99 * 1024) return PyErr_Format(PyExc_ValueError, "head_dim %d too large", hd);
        static size_t tiled_attr = 0;
        if (tsmem > 48 * 1024 && tsmem > tiled_attr) {
            cudaError_t err = cudaFuncSetAttribute(attention_tiled_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   static_cast<int>(tsmem));
            if (err != cudaSuccess) return cuda_error("attention smem attribute", err);
            tiled_attr = tsmem;
        }
        attention_tiled_kernel<T><<<dim3(heads, (ntok + kAttnTQ - 1) / kAttnTQ), hd, tsmem>>>(
            q, kc, vc, o, heads, kv_heads, hd, A, seq_len, ntok, scale, bidirectional, window);
        return launch_result("attention");
    }
    if (smem > 48 * 1024) {
        cudaError_t err = cudaFuncSetAttribute(attention_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        if (err != cudaSuccess) return cuda_error("attention smem attribute", err);
    }
    attention_kernel<T><<<dim3(heads, ntok), 256, smem>>>(q, kc, vc, o, heads, kv_heads, hd, A, seq_len, scale,
                                                          bidirectional, window);
    return launch_result("attention");
}

PyObject *Runtime_attention(RuntimeObject *self, PyObject *args) {
    const char *qname, *kcname, *vcname, *oname;
    int heads = 0, kv_heads = 0, hd = 0, max_seq = 0, seq_len = 0, ntok = 1, bidirectional = 0, window = 0;
    if (!PyArg_ParseTuple(args, "ssssiiiii|iii", &qname, &kcname, &vcname, &oname, &heads, &kv_heads, &hd, &max_seq,
                          &seq_len, &ntok, &bidirectional, &window))
        return nullptr;
    if (kv_heads <= 0 || heads % kv_heads != 0) return PyErr_Format(PyExc_ValueError, "heads must be divisible by kv_heads");
    // seq_len is the causal length of the first query token; token t sees seq_len + t positions.
    if (ntok < 1 || seq_len <= 0 || seq_len + ntok - 1 > max_seq)
        return PyErr_Format(PyExc_ValueError, "invalid seq_len %d for %d tokens", seq_len, ntok);
    const size_t n = static_cast<size_t>(heads) * hd * ntok, cache = static_cast<size_t>(kv_heads) * hd * max_seq;
    int ke = 0, ve = 0;
    const float *q = find_buffer(self, qname, n);
    const void *kc = q ? find_kv_buffer(self, kcname, cache, &ke) : nullptr;
    const void *vc = kc ? find_kv_buffer(self, vcname, cache, &ve) : nullptr;
    float *o = vc ? find_buffer(self, oname, n) : nullptr;
    if (o == nullptr) return nullptr;
    if (ke != ve) return PyErr_Format(PyExc_TypeError, "K and V caches differ in precision");
    const bool prompt_pass = self->gemm_min_rows > 0 && ntok >= self->gemm_min_rows;
    if (ke == 2)
        return launch_attention(static_cast<const __half *>(kc), static_cast<const __half *>(vc), q, o, heads, kv_heads, hd,
                                contiguous_kv(max_seq), seq_len, ntok, bidirectional, window, prompt_pass);
    return launch_attention(static_cast<const float *>(kc), static_cast<const float *>(vc), q, o, heads, kv_heads, hd,
                            contiguous_kv(max_seq), seq_len, ntok, bidirectional, window, prompt_pass);
}

PyObject *Runtime_conv_update(RuntimeObject *self, PyObject *args) {
    const char *xname, *sname, *wname, *oname, *snapname = nullptr;
    int channels = 0, K = 0, ntok = 1;
    if (!PyArg_ParseTuple(args, "ssssii|iz", &xname, &sname, &wname, &oname, &channels, &K, &ntok, &snapname))
        return nullptr;
    if (ntok < 1) return PyErr_Format(PyExc_ValueError, "ntok must be positive");
    const float *x = find_buffer(self, xname, static_cast<size_t>(channels) * ntok);
    float *st = x ? find_buffer(self, sname, static_cast<size_t>(channels) * K) : nullptr;
    float *o = st ? find_buffer(self, oname, static_cast<size_t>(channels) * ntok) : nullptr;
    const float *w = o ? find_f32_weight(self, wname, static_cast<size_t>(channels) * K) : nullptr;
    if (w == nullptr) return nullptr;
    float *snap = nullptr;
    if (snapname != nullptr) {
        snap = find_buffer(self, snapname, static_cast<size_t>(channels) * K * (ntok - 1));
        if (snap == nullptr) return nullptr;
    }
    if (K > kConvMaxK) return PyErr_Format(PyExc_ValueError, "conv kernel size %d > %d", K, kConvMaxK);
    conv_update_kernel<<<blocks_for(channels, 256), 256>>>(x, st, w, o, channels, K, ntok, snap, 1);
    return launch_result("conv_update");
}

// write_state = 0 leaves the state untouched (speculative verification); `scratch` (value_heads*kd*vd floats)
// is needed then by the fallback kernel, which updates its state in place.
void launch_gated_delta(const float *conv, const float *b, const float *a, const float *ssm_a, const float *dt, float *st,
                        float *o, int key_heads, int value_heads, int kd, int vd, float eps, int head_order, int ntok,
                        float *snap, int write_state = 1, float *scratch = nullptr) {
    if (vd % 32 == 0 && (kd == 32 || kd == 64 || kd == 128 || kd == 256)) {
        const dim3 grid(value_heads, vd / 32);
        const int w = write_state;
        switch (kd) {
            case 32: gated_delta_reg_kernel<8><<<grid, 32 * kGdP>>>(conv, b, a, ssm_a, dt, st, o, key_heads, value_heads, vd, eps, head_order, ntok, snap, w); break;
            case 64: gated_delta_reg_kernel<16><<<grid, 32 * kGdP>>>(conv, b, a, ssm_a, dt, st, o, key_heads, value_heads, vd, eps, head_order, ntok, snap, w); break;
            case 128: gated_delta_reg_kernel<32><<<grid, 32 * kGdP>>>(conv, b, a, ssm_a, dt, st, o, key_heads, value_heads, vd, eps, head_order, ntok, snap, w); break;
            default: gated_delta_reg_kernel<64><<<grid, 32 * kGdP>>>(conv, b, a, ssm_a, dt, st, o, key_heads, value_heads, vd, eps, head_order, ntok, snap, w); break;
        }
        return;
    }
    if (!write_state) {  // fallback kernel works in place: run it on a copy
        cudaMemcpyAsync(scratch, st, static_cast<size_t>(value_heads) * kd * vd * sizeof(float), cudaMemcpyDeviceToDevice, 0);
        st = scratch;
    }
    const int threads = vd < 32 ? 32 : ((vd + 31) / 32) * 32;
    gated_delta_kernel<<<value_heads, threads, 2 * kd * sizeof(float)>>>(conv, b, a, ssm_a, dt, st, o, key_heads,
                                                                         value_heads, kd, vd, eps, head_order, ntok, snap);
}

PyObject *Runtime_gated_delta(RuntimeObject *self, PyObject *args) {
    const char *cname, *bname, *aname, *ssm_a_name, *dt_name, *sname, *oname, *snapname = nullptr;
    int key_heads = 0, value_heads = 0, kd = 0, vd = 0, head_order = 0, ntok = 1;
    float eps = 1e-6f;
    if (!PyArg_ParseTuple(args, "sssssssiiiifi|iz", &cname, &bname, &aname, &ssm_a_name, &dt_name, &sname, &oname,
                          &key_heads, &value_heads, &kd, &vd, &eps, &head_order, &ntok, &snapname))
        return nullptr;
    if (key_heads <= 0 || value_heads % key_heads != 0) return PyErr_Format(PyExc_ValueError, "value heads must be a multiple of key heads");
    if (ntok < 1) return PyErr_Format(PyExc_ValueError, "ntok must be positive");
    const size_t conv_dim = static_cast<size_t>(2) * key_heads * kd + static_cast<size_t>(value_heads) * vd;
    const size_t state_n = static_cast<size_t>(value_heads) * kd * vd;
    const float *conv = find_buffer(self, cname, conv_dim * ntok);
    const float *b = conv ? find_buffer(self, bname, static_cast<size_t>(value_heads) * ntok) : nullptr;
    const float *a = b ? find_buffer(self, aname, static_cast<size_t>(value_heads) * ntok) : nullptr;
    float *st = a ? find_buffer(self, sname, state_n) : nullptr;
    float *o = st ? find_buffer(self, oname, static_cast<size_t>(value_heads) * vd * ntok) : nullptr;
    const float *ssm_a = o ? find_f32_weight(self, ssm_a_name, value_heads) : nullptr;
    const float *dt = ssm_a ? find_f32_weight(self, dt_name, value_heads) : nullptr;
    if (dt == nullptr) return nullptr;
    float *snap = nullptr;
    if (snapname != nullptr) {
        snap = find_buffer(self, snapname, state_n * (ntok - 1));
        if (snap == nullptr) return nullptr;
    }
    launch_gated_delta(conv, b, a, ssm_a, dt, st, o, key_heads, value_heads, kd, vd, eps, head_order, ntok, snap);
    return launch_result("gated_delta");
}

PyObject *Runtime_argmax(RuntimeObject *self, PyObject *args) {
    const char *xname;
    int n = 0;
    if (!PyArg_ParseTuple(args, "si", &xname, &n)) return nullptr;
    const float *x = find_buffer(self, xname, n);
    if (x == nullptr) return nullptr;
    argmax_kernel<<<1, 1024>>>(x, n, self->argmax_out);
    int host = -1;
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaMemcpy(&host, self->argmax_out, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return cuda_error("argmax", err);
    return PyLong_FromLong(host);
}

// argmax_rows(x, n, ntok) -> [int] * ntok (row-wise; only the indices leave the GPU).
PyObject *Runtime_argmax_rows(RuntimeObject *self, PyObject *args) {
    const char *xname;
    int n = 0, ntok = 1;
    if (!PyArg_ParseTuple(args, "sii", &xname, &n, &ntok)) return nullptr;
    if (ntok < 1 || ntok > kMaxArgmaxRows) return PyErr_Format(PyExc_ValueError, "ntok must be in [1, %d]", kMaxArgmaxRows);
    const float *x = find_buffer(self, xname, static_cast<size_t>(n) * ntok);
    if (x == nullptr) return nullptr;
    argmax_kernel<<<ntok, 1024>>>(x, n, self->argmax_out);
    int host[kMaxArgmaxRows];
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaMemcpy(host, self->argmax_out, ntok * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return cuda_error("argmax_rows", err);
    PyObject *list = PyList_New(ntok);
    for (int t = 0; t < ntok; ++t) PyList_SET_ITEM(list, t, PyLong_FromLong(host[t]));
    return list;
}

// copy(dst, dst_offset, src, src_offset, n): device-to-device float copy between buffers.
PyObject *Runtime_copy(RuntimeObject *self, PyObject *args) {
    const char *dname, *sname;
    Py_ssize_t doff = 0, soff = 0, n = 0;
    if (!PyArg_ParseTuple(args, "snsnn", &dname, &doff, &sname, &soff, &n)) return nullptr;
    if (doff < 0 || soff < 0 || n < 0) return PyErr_Format(PyExc_ValueError, "offsets and count must be non-negative");
    // Offsets and count in elements; both buffers must share the element size (fp32, or fp16 KV pools).
    DeviceBuffer *db = find_any_buffer(self, dname);
    DeviceBuffer *sb = db ? find_any_buffer(self, sname) : nullptr;
    if (sb == nullptr) return nullptr;
    if (db->elem != sb->elem) return PyErr_Format(PyExc_TypeError, "copy between buffers of different precision");
    if (static_cast<size_t>(doff + n) > db->n || static_cast<size_t>(soff + n) > sb->n)
        return PyErr_Format(PyExc_ValueError, "copy out of bounds (%s <- %s)", dname, sname);
    const size_t e = static_cast<size_t>(db->elem);
    cudaError_t err = cudaMemcpyAsync(reinterpret_cast<char *>(db->data) + doff * e, reinterpret_cast<char *>(sb->data) + soff * e,
                                      n * e, cudaMemcpyDeviceToDevice, 0);
    if (err != cudaSuccess) return cuda_error("buffer copy", err);
    Py_RETURN_NONE;
}

// dyn_conv(x, dyn, base_weight, base_offset, out, ntok, hidden, K, group, sel): DFlash 2 convolution.
// base_offset selects base_kernel[sel] inside the F32 [2][K][hidden] weight.
PyObject *Runtime_dyn_conv(RuntimeObject *self, PyObject *args) {
    const char *xname, *dname, *bname, *oname;
    int ntok = 0, hidden = 0, K = 0, group = 0, sel = 0;
    if (!PyArg_ParseTuple(args, "ssssiiiii", &xname, &dname, &bname, &oname, &ntok, &hidden, &K, &group, &sel)) return nullptr;
    if (ntok < 1 || hidden % group != 0 || sel < 0 || sel > 1) return PyErr_Format(PyExc_ValueError, "invalid dyn_conv shape");
    const size_t n = static_cast<size_t>(ntok) * hidden;
    const float *x = find_buffer(self, xname, n);
    const float *dyn = x ? find_buffer(self, dname, static_cast<size_t>(ntok) * 2 * K * (hidden / group)) : nullptr;
    float *o = dyn ? find_buffer(self, oname, n) : nullptr;
    const float *base = o ? find_f32_weight(self, bname, static_cast<size_t>(2) * K * hidden) : nullptr;
    if (base == nullptr) return nullptr;
    dyn_conv_kernel<<<blocks_for(n, 256), 256>>>(x, dyn, base + static_cast<size_t>(sel) * K * hidden, o, ntok, hidden, K, group, sel);
    return launch_result("dyn_conv");
}

// topk_rows(x, n, ntok, k) -> (indices [ntok*k], values [ntok*k]) as lists (k <= 32).
PyObject *Runtime_topk_rows(RuntimeObject *self, PyObject *args) {
    const char *xname;
    int n = 0, ntok = 1, k = 1;
    if (!PyArg_ParseTuple(args, "siii", &xname, &n, &ntok, &k)) return nullptr;
    if (ntok < 1 || ntok > kMaxTokens || k < 1 || k > 32) return PyErr_Format(PyExc_ValueError, "invalid topk shape");
    const float *x = find_buffer(self, xname, static_cast<size_t>(n) * ntok);
    if (x == nullptr) return nullptr;
    int *didx = nullptr;
    float *dval = nullptr;
    cudaError_t err = cudaMalloc(&didx, ntok * k * sizeof(int));
    if (err == cudaSuccess) err = cudaMalloc(&dval, ntok * k * sizeof(float));
    if (err == cudaSuccess) {
        topk_kernel<<<ntok, 1024>>>(x, n, k, didx, dval);
        err = cudaGetLastError();
    }
    int hidx[kMaxTokens * 32];
    float hval[kMaxTokens * 32];
    if (err == cudaSuccess) err = cudaMemcpy(hidx, didx, ntok * k * sizeof(int), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(hval, dval, ntok * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(didx);
    cudaFree(dval);
    if (err != cudaSuccess) return cuda_error("topk_rows", err);
    PyObject *li = PyList_New(ntok * k), *lv = PyList_New(ntok * k);
    for (int i = 0; i < ntok * k; ++i) {
        PyList_SET_ITEM(li, i, PyLong_FromLong(hidx[i]));
        PyList_SET_ITEM(lv, i, PyFloat_FromDouble(hval[i]));
    }
    return Py_BuildValue("(NN)", li, lv);
}

PyObject *Runtime_synchronize(RuntimeObject *, PyObject *) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return cuda_error("synchronize", err);
    Py_RETURN_NONE;
}

PyObject *Runtime_mem_info(RuntimeObject *, PyObject *) {
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) return cuda_error("cudaMemGetInfo", err);
    return Py_BuildValue("(nn)", static_cast<Py_ssize_t>(free_bytes), static_cast<Py_ssize_t>(total_bytes));
}

// matvec(name, x: float32 buffer) -> bytes (float32 y). Debug/bring-up entry point.
PyObject *Runtime_matvec(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    Py_buffer x;
    if (!PyArg_ParseTuple(args, "sy*", &name, &x)) return nullptr;
    auto found = self->tensors->find(name);
    if (found == self->tensors->end()) {
        PyBuffer_Release(&x);
        return PyErr_Format(PyExc_KeyError, "%s is not resident on the GPU", name);
    }
    const DeviceTensor &dt = found->second;
    if (static_cast<size_t>(x.len) != static_cast<size_t>(dt.cols) * sizeof(float)) {
        PyBuffer_Release(&x);
        return PyErr_Format(PyExc_ValueError, "%s: x has %zd bytes, expected %d float32", name, x.len, dt.cols);
    }
    if (!ensure_scratch(self, dt.cols, dt.rows)) {
        PyBuffer_Release(&x);
        return cuda_error("scratch allocation", cudaErrorMemoryAllocation);
    }
    cudaError_t err = cudaMemcpy(self->scratch_x, x.buf, x.len, cudaMemcpyHostToDevice);
    PyBuffer_Release(&x);
    TypeTraits t;
    type_traits(dt.type, &t);
    if (err == cudaSuccess)
        err = launch_qmatvec(dt.type, dt.data, self->scratch_x, self->scratch_y, dt.rows, dt.cols, dt.row_bytes, t);
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return cuda_error("qmatvec", err);
    PyObject *out = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(dt.rows) * sizeof(float));
    if (out == nullptr) return nullptr;
    err = cudaMemcpy(PyBytes_AS_STRING(out), self->scratch_y, dt.rows * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        Py_DECREF(out);
        return cuda_error("qmatvec D2H", err);
    }
    return out;
}

PyObject *Runtime_buffer_elem(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    if (!PyArg_ParseTuple(args, "s", &name)) return nullptr;
    DeviceBuffer *b = find_any_buffer(self, name);
    return b == nullptr ? nullptr : PyLong_FromLong(b->elem);
}

PyObject *Runtime_set_gemm_min_rows(RuntimeObject *self, PyObject *args) {
    int n = 0;
    if (!PyArg_ParseTuple(args, "i", &n)) return nullptr;
    if (n < 0) return PyErr_Format(PyExc_ValueError, "gemm_min_rows must be >= 0");
    self->gemm_min_rows = n;
    Py_RETURN_NONE;
}

// ---- multi-sequence passes ------------------------------------------------------------------------

// set_paging(page_table_buffer, page_size, pages_per_seq, max_slots): KV caches used by the *_seg ops are
// page pools [page][kv_head][page_size][hd]; the int32 page table holds [max_slots][pages_per_seq] page ids.
PyObject *Runtime_set_paging(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    int page = 0, pps = 0, slots = 0;
    if (!PyArg_ParseTuple(args, "siii", &name, &page, &pps, &slots)) return nullptr;
    if (page < 1 || pps < 1 || slots < 1) return PyErr_Format(PyExc_ValueError, "invalid paging shape");
    const float *pt = find_buffer(self, name, static_cast<size_t>(pps) * slots);
    if (pt == nullptr) return nullptr;
    self->page_table = reinterpret_cast<const int *>(pt);
    self->page_size = page;
    self->pages_per_seq = pps;
    self->max_slots = slots;
    Py_RETURN_NONE;
}

// set_segments([(slot, pos0, rows), ...]) -> total rows: row layout of the next *_seg ops (rows of a
// segment are consecutive positions of one sequence slot; segments are stacked in order).
PyObject *Runtime_set_segments(RuntimeObject *self, PyObject *args) {
    PyObject *list = nullptr;
    if (!PyArg_ParseTuple(args, "O", &list)) return nullptr;
    if (self->page_table == nullptr) return PyErr_Format(PyExc_RuntimeError, "set_paging first");
    PyObject *seq = PySequence_Fast(list, "segments must be a sequence");
    if (seq == nullptr) return nullptr;
    if (self->seg == nullptr) self->seg = new std::vector<int>();
    self->seg->clear();
    int row0 = 0;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        int slot = 0, pos0 = 0, rows = 0, r0 = -1;
        if (!PyArg_ParseTuple(PySequence_Fast_GET_ITEM(seq, i), "iii|i", &slot, &pos0, &rows, &r0)) {
            Py_DECREF(seq);
            return nullptr;
        }
        if (r0 >= 0) row0 = r0;  // explicit row offset (commit replays rows of an earlier pass)
        if (slot < 0 || slot >= self->max_slots || pos0 < 0 || rows < 1 ||
            pos0 + rows > self->page_size * self->pages_per_seq) {
            Py_DECREF(seq);
            return PyErr_Format(PyExc_ValueError, "segment (%d, %d, %d) outside the paged context", slot, pos0, rows);
        }
        self->seg->insert(self->seg->end(), {slot, pos0, rows, row0});
        row0 += rows;
    }
    Py_DECREF(seq);
    if (self->seg->empty()) return PyErr_Format(PyExc_ValueError, "no segments");
    return PyLong_FromLong(row0);
}

static int seg_rows(RuntimeObject *self) {  // rows spanned by the segments (max row0 + rows)
    const auto &g = *self->seg;
    int n = 0;
    for (size_t i = 0; i < g.size(); i += 4) n = g[i + 2] + g[i + 3] > n ? g[i + 2] + g[i + 3] : n;
    return n;
}

static bool seg_ready(RuntimeObject *self) {
    if (self->seg == nullptr || self->seg->empty() || self->page_table == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "set_paging and set_segments must be called before *_seg ops");
        return false;
    }
    return true;
}

static KvAddr seg_kv(RuntimeObject *self, int slot, int kv_heads) {
    return KvAddr{self->page_table + static_cast<size_t>(slot) * self->pages_per_seq,
                  self->page_size * self->pages_per_seq, self->page_size, kv_heads};
}

// rope_seg(x, heads, hd, rot, base): NeoX RoPE with each segment's positions.
PyObject *Runtime_rope_seg(RuntimeObject *self, PyObject *args) {
    const char *xname;
    int heads = 0, hd = 0, rot = 0;
    double base = 10000.0;
    if (!PyArg_ParseTuple(args, "siiid", &xname, &heads, &hd, &rot, &base)) return nullptr;
    if (!seg_ready(self)) return nullptr;
    float *x = find_buffer(self, xname, static_cast<size_t>(heads) * hd * seg_rows(self));
    if (x == nullptr) return nullptr;
    const auto &g = *self->seg;
    for (size_t i = 0; i < g.size(); i += 4) {
        const int pos0 = g[i + 1], rows = g[i + 2], row0 = g[i + 3];
        rope_neox_kernel<<<blocks_for(static_cast<size_t>(rows) * heads * rot / 2, 128), 128>>>(
            x + static_cast<size_t>(row0) * heads * hd, rows, heads, hd, rot, pos0, base);
    }
    return launch_result("rope_seg");
}

// kv_append_seg(k, v, kc, vc, kv_heads, hd): append each segment's K/V rows to its sequence's pages.
PyObject *Runtime_kv_append_seg(RuntimeObject *self, PyObject *args) {
    const char *kname, *vname, *kcname, *vcname;
    int kv_heads = 0, hd = 0;
    if (!PyArg_ParseTuple(args, "ssssii", &kname, &vname, &kcname, &vcname, &kv_heads, &hd)) return nullptr;
    if (!seg_ready(self)) return nullptr;
    const size_t rowsz = static_cast<size_t>(kv_heads) * hd;
    int ke = 0, ve = 0;
    const float *k = find_buffer(self, kname, rowsz * seg_rows(self));
    const float *v = k ? find_buffer(self, vname, rowsz * seg_rows(self)) : nullptr;
    void *kc = v ? find_kv_buffer(self, kcname, rowsz, &ke) : nullptr;
    void *vc = kc ? find_kv_buffer(self, vcname, rowsz, &ve) : nullptr;
    if (vc == nullptr) return nullptr;
    if (ke != ve) return PyErr_Format(PyExc_TypeError, "K and V caches differ in precision");
    const auto &g = *self->seg;
    for (size_t i = 0; i < g.size(); i += 4) {
        const int slot = g[i], pos0 = g[i + 1], rows = g[i + 2], row0 = g[i + 3];
        const size_t n = rowsz * rows;
        const KvAddr A = seg_kv(self, slot, kv_heads);
        if (ke == 2)
            kv_append_kernel<__half><<<blocks_for(n, 256), 256>>>(k + row0 * rowsz, v + row0 * rowsz, static_cast<__half *>(kc),
                                                                  static_cast<__half *>(vc), rows, kv_heads, A, hd, pos0);
        else
            kv_append_kernel<float><<<blocks_for(n, 256), 256>>>(k + row0 * rowsz, v + row0 * rowsz, static_cast<float *>(kc),
                                                                 static_cast<float *>(vc), rows, kv_heads, A, hd, pos0);
    }
    return launch_result("kv_append_seg");
}

// attention_seg(q, kc, vc, out, heads, kv_heads, hd): causal attention of each segment over its own pages.
PyObject *Runtime_attention_seg(RuntimeObject *self, PyObject *args) {
    const char *qname, *kcname, *vcname, *oname;
    int heads = 0, kv_heads = 0, hd = 0;
    if (!PyArg_ParseTuple(args, "ssssiii", &qname, &kcname, &vcname, &oname, &heads, &kv_heads, &hd)) return nullptr;
    if (!seg_ready(self)) return nullptr;
    if (kv_heads <= 0 || heads % kv_heads != 0) return PyErr_Format(PyExc_ValueError, "heads must be divisible by kv_heads");
    const size_t rowsz = static_cast<size_t>(heads) * hd;
    int ke = 0, ve = 0;
    const float *q = find_buffer(self, qname, rowsz * seg_rows(self));
    const void *kc = q ? find_kv_buffer(self, kcname, 1, &ke) : nullptr;
    const void *vc = kc ? find_kv_buffer(self, vcname, 1, &ve) : nullptr;
    float *o = vc ? find_buffer(self, oname, rowsz * seg_rows(self)) : nullptr;
    if (o == nullptr) return nullptr;
    if (ke != ve) return PyErr_Format(PyExc_TypeError, "K and V caches differ in precision");
    const auto &g = *self->seg;
    for (size_t i = 0; i < g.size(); i += 4) {
        const int slot = g[i], pos0 = g[i + 1], rows = g[i + 2], row0 = g[i + 3];
        const bool prompt_pass = self->gemm_min_rows > 0 && rows >= self->gemm_min_rows;
        const KvAddr A = seg_kv(self, slot, kv_heads);
        PyObject *r = ke == 2
            ? launch_attention(static_cast<const __half *>(kc), static_cast<const __half *>(vc), q + row0 * rowsz,
                               o + row0 * rowsz, heads, kv_heads, hd, A, pos0 + 1, rows, 0, 0, prompt_pass)
            : launch_attention(static_cast<const float *>(kc), static_cast<const float *>(vc), q + row0 * rowsz,
                               o + row0 * rowsz, heads, kv_heads, hd, A, pos0 + 1, rows, 0, 0, prompt_pass);
        if (r == nullptr) return nullptr;
        Py_DECREF(r);
    }
    Py_RETURN_NONE;
}

// conv_update_seg(x, state, w, out, channels, K[, snap]): per-segment causal conv over the slot's state
// ([slots][channels][K]); snapshots need a single segment.
PyObject *Runtime_conv_update_seg(RuntimeObject *self, PyObject *args) {
    const char *xname, *sname, *wname, *oname, *snapname = nullptr;
    int channels = 0, K = 0, write_state = 1;
    if (!PyArg_ParseTuple(args, "ssssii|zi", &xname, &sname, &wname, &oname, &channels, &K, &snapname, &write_state))
        return nullptr;
    if (!seg_ready(self)) return nullptr;
    if (K > kConvMaxK) return PyErr_Format(PyExc_ValueError, "conv kernel size %d > %d", K, kConvMaxK);
    const int ntok = seg_rows(self);
    const size_t stride = static_cast<size_t>(channels) * K;
    const float *x = find_buffer(self, xname, static_cast<size_t>(channels) * ntok);
    float *st = x ? find_buffer(self, sname, stride * self->max_slots) : nullptr;
    float *o = st ? find_buffer(self, oname, static_cast<size_t>(channels) * ntok) : nullptr;
    const float *w = o ? find_f32_weight(self, wname, stride) : nullptr;
    if (w == nullptr) return nullptr;
    const auto &g = *self->seg;
    float *snap = nullptr;
    if (snapname != nullptr) {
        if (g.size() != 4) return PyErr_Format(PyExc_ValueError, "snapshots need a single-segment pass");
        snap = find_buffer(self, snapname, stride * (ntok - 1));
        if (snap == nullptr) return nullptr;
    }
    for (size_t i = 0; i < g.size(); i += 4) {
        const int slot = g[i], rows = g[i + 2], row0 = g[i + 3];
        conv_update_kernel<<<blocks_for(channels, 256), 256>>>(x + static_cast<size_t>(row0) * channels, st + slot * stride, w,
                                                               o + static_cast<size_t>(row0) * channels, channels, K, rows, snap,
                                                               write_state);
    }
    return launch_result("conv_update_seg");
}

// gated_delta_seg(conv, beta, alpha, ssm_a, dt_bias, state, out, kh, vh, kd, vd, eps, head_order[, snap]):
// per-segment gated delta over the slot's state ([slots][vh][kd][vd]).
PyObject *Runtime_gated_delta_seg(RuntimeObject *self, PyObject *args) {
    const char *cname, *bname, *aname, *ssm_a_name, *dt_name, *sname, *oname, *snapname = nullptr;
    int key_heads = 0, value_heads = 0, kd = 0, vd = 0, head_order = 0, write_state = 1;
    float eps = 1e-6f;
    if (!PyArg_ParseTuple(args, "sssssssiiiifi|zi", &cname, &bname, &aname, &ssm_a_name, &dt_name, &sname, &oname,
                          &key_heads, &value_heads, &kd, &vd, &eps, &head_order, &snapname, &write_state))
        return nullptr;
    if (!seg_ready(self)) return nullptr;
    if (key_heads <= 0 || value_heads % key_heads != 0) return PyErr_Format(PyExc_ValueError, "value heads must be a multiple of key heads");
    const int ntok = seg_rows(self);
    const size_t conv_dim = static_cast<size_t>(2) * key_heads * kd + static_cast<size_t>(value_heads) * vd;
    const size_t state_n = static_cast<size_t>(value_heads) * kd * vd;
    const float *conv = find_buffer(self, cname, conv_dim * ntok);
    const float *b = conv ? find_buffer(self, bname, static_cast<size_t>(value_heads) * ntok) : nullptr;
    const float *a = b ? find_buffer(self, aname, static_cast<size_t>(value_heads) * ntok) : nullptr;
    float *st = a ? find_buffer(self, sname, state_n * self->max_slots) : nullptr;
    float *o = st ? find_buffer(self, oname, static_cast<size_t>(value_heads) * vd * ntok) : nullptr;
    const float *ssm_a = o ? find_f32_weight(self, ssm_a_name, value_heads) : nullptr;
    const float *dt = ssm_a ? find_f32_weight(self, dt_name, value_heads) : nullptr;
    if (dt == nullptr) return nullptr;
    const auto &g = *self->seg;
    float *snap = nullptr;
    if (snapname != nullptr) {
        if (g.size() != 4) return PyErr_Format(PyExc_ValueError, "snapshots need a single-segment pass");
        snap = find_buffer(self, snapname, state_n * (ntok - 1));
        if (snap == nullptr) return nullptr;
    }
    if (!write_state && !ensure_scratch(self, state_n, 0)) return cuda_error("gated_delta_seg scratch", cudaErrorMemoryAllocation);
    for (size_t i = 0; i < g.size(); i += 4) {
        const int slot = g[i], rows = g[i + 2], row0 = g[i + 3];
        launch_gated_delta(conv + row0 * conv_dim, b + static_cast<size_t>(row0) * value_heads,
                           a + static_cast<size_t>(row0) * value_heads, ssm_a, dt, st + slot * state_n,
                           o + static_cast<size_t>(row0) * value_heads * vd, key_heads, value_heads, kd, vd, eps,
                           head_order, rows, snap, write_state, self->scratch_x);
    }
    return launch_result("gated_delta_seg");
}

// zero_range(name, offset, n): clear n elements from offset.
PyObject *Runtime_zero_range(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    Py_ssize_t off = 0, n = 0;
    if (!PyArg_ParseTuple(args, "snn", &name, &off, &n)) return nullptr;
    DeviceBuffer *b = find_any_buffer(self, name);
    if (b == nullptr) return nullptr;
    if (off < 0 || n < 0 || static_cast<size_t>(off + n) > b->n) return PyErr_Format(PyExc_ValueError, "zero_range out of bounds");
    cudaError_t err = cudaMemset(reinterpret_cast<char *>(b->data) + static_cast<size_t>(off) * b->elem, 0,
                                 static_cast<size_t>(n) * b->elem);
    if (err != cudaSuccess) return cuda_error("cudaMemset", err);
    Py_RETURN_NONE;
}

PyMethodDef Runtime_methods[] = {
    {"set_paging", reinterpret_cast<PyCFunction>(Runtime_set_paging), METH_VARARGS,
     "set_paging(page_table, page_size, pages_per_seq, max_slots): paged KV pools for the *_seg ops."},
    {"set_segments", reinterpret_cast<PyCFunction>(Runtime_set_segments), METH_VARARGS,
     "set_segments([(slot, pos0, rows), ...]) -> total rows: row layout of the next *_seg ops."},
    {"rope_seg", reinterpret_cast<PyCFunction>(Runtime_rope_seg), METH_VARARGS, "rope_seg(x, heads, hd, rot, base)."},
    {"kv_append_seg", reinterpret_cast<PyCFunction>(Runtime_kv_append_seg), METH_VARARGS, "kv_append_seg(k, v, kc, vc, kv_heads, hd)."},
    {"attention_seg", reinterpret_cast<PyCFunction>(Runtime_attention_seg), METH_VARARGS,
     "attention_seg(q, kc, vc, out, heads, kv_heads, hd): causal paged attention per segment."},
    {"conv_update_seg", reinterpret_cast<PyCFunction>(Runtime_conv_update_seg), METH_VARARGS,
     "conv_update_seg(x, state, w, out, channels, K[, snap])."},
    {"gated_delta_seg", reinterpret_cast<PyCFunction>(Runtime_gated_delta_seg), METH_VARARGS,
     "gated_delta_seg(conv, beta, alpha, ssm_a, dt_bias, state, out, kh, vh, kd, vd, eps, head_order[, snap])."},
    {"zero_range", reinterpret_cast<PyCFunction>(Runtime_zero_range), METH_VARARGS, "zero_range(name, offset, n)."},
    {"buffer_elem", reinterpret_cast<PyCFunction>(Runtime_buffer_elem), METH_VARARGS,
     "buffer_elem(name) -> bytes per element (4 = fp32, 2 = fp16)."},
    {"set_gemm_min_rows", reinterpret_cast<PyCFunction>(Runtime_set_gemm_min_rows), METH_VARARGS,
     "set_gemm_min_rows(n): passes of n or more rows use the prompt kernels (tensor-core GEMM, tiled attention); 0 = more than 8."},
    {"upload", reinterpret_cast<PyCFunction>(Runtime_upload), METH_VARARGS,
     "upload(name, raw_bytes, gguf_type_id, cols, rows): copy raw GGUF blocks to VRAM."},
    {"free", reinterpret_cast<PyCFunction>(Runtime_free), METH_VARARGS, "free(name): release a device tensor."},
    {"has", reinterpret_cast<PyCFunction>(Runtime_has), METH_VARARGS, "has(name) -> bool."},
    {"device_bytes", reinterpret_cast<PyCFunction>(Runtime_device_bytes), METH_NOARGS,
     "Total bytes of resident weight tensors."},
    {"mem_info", reinterpret_cast<PyCFunction>(Runtime_mem_info), METH_NOARGS, "(free_bytes, total_bytes) of the device."},
    {"pinned_bytes", reinterpret_cast<PyCFunction>(Runtime_pinned_bytes), METH_NOARGS, "Bytes of streamed (pinned host) tensors."},
    {"streamed_bytes", reinterpret_cast<PyCFunction>(Runtime_streamed_bytes), METH_NOARGS, "Total bytes streamed host->device so far."},
    {"staging_bytes", reinterpret_cast<PyCFunction>(Runtime_staging_bytes), METH_NOARGS, "Bytes of streaming staging buffers."},
    {"set_arena", reinterpret_cast<PyCFunction>(Runtime_set_arena), METH_VARARGS, "set_arena(nbytes): weight arena for resident uploads."},
    {"tensor_info", reinterpret_cast<PyCFunction>(Runtime_tensor_info), METH_VARARGS,
     "tensor_info(name) -> (address, type, rows, cols, row_bytes, nbytes, resident)."},
    {"buffer_ptr", reinterpret_cast<PyCFunction>(Runtime_buffer_ptr), METH_VARARGS, "buffer_ptr(name) -> (address, n)."},
    {"set_staging", reinterpret_cast<PyCFunction>(Runtime_set_staging), METH_VARARGS, "set_staging(nbytes): device staging buffer for streamed tensors."},
    {"set_stream_order", reinterpret_cast<PyCFunction>(Runtime_set_stream_order), METH_VARARGS,
     "set_stream_order(names, slots): prefetch streamed tensors in this cyclic use order into `slots` buffers."},
    {"alloc", reinterpret_cast<PyCFunction>(Runtime_alloc), METH_VARARGS, "alloc(name, n): zeroed float32 device buffer."},
    {"zero", reinterpret_cast<PyCFunction>(Runtime_zero), METH_VARARGS, "zero(name)."},
    {"buffer_bytes", reinterpret_cast<PyCFunction>(Runtime_buffer_bytes), METH_NOARGS, "Total bytes of device buffers."},
    {"write", reinterpret_cast<PyCFunction>(Runtime_write), METH_VARARGS, "write(name, f32_bytes)."},
    {"read", reinterpret_cast<PyCFunction>(Runtime_read), METH_VARARGS, "read(name[, n]) -> f32 bytes (debug/explicit transfers)."},
    {"qmv", reinterpret_cast<PyCFunction>(Runtime_qmv), METH_VARARGS, "qmv(weight, x, y): quantized matvec on device buffers."},
    {"rmsnorm", reinterpret_cast<PyCFunction>(Runtime_rmsnorm), METH_VARARGS, "rmsnorm(x, w, y, n, eps[, groups])."},
    {"gated_rmsnorm", reinterpret_cast<PyCFunction>(Runtime_gated_rmsnorm), METH_VARARGS, "gated_rmsnorm(x, w, z, y, n, eps, groups)."},
    {"add", reinterpret_cast<PyCFunction>(Runtime_add), METH_VARARGS, "add(a, b, out, n)."},
    {"silu_mul", reinterpret_cast<PyCFunction>(Runtime_silu_mul), METH_VARARGS, "silu_mul(gate, up, out, n)."},
    {"sigmoid_mul", reinterpret_cast<PyCFunction>(Runtime_sigmoid_mul), METH_VARARGS, "sigmoid_mul(x, gate, out, n)."},
    {"split_gated_q", reinterpret_cast<PyCFunction>(Runtime_split_gated_q), METH_VARARGS, "split_gated_q(raw, q, gate, heads, hd)."},
    {"rope", reinterpret_cast<PyCFunction>(Runtime_rope), METH_VARARGS, "rope(x, heads, hd, rot, pos, base): NeoX partial RoPE in place."},
    {"kv_append", reinterpret_cast<PyCFunction>(Runtime_kv_append), METH_VARARGS, "kv_append(k, v, kc, vc, kv_heads, max_seq, hd, pos)."},
    {"attention", reinterpret_cast<PyCFunction>(Runtime_attention), METH_VARARGS, "attention(q, kc, vc, out, heads, kv_heads, hd, max_seq, seq_len)."},
    {"conv_update", reinterpret_cast<PyCFunction>(Runtime_conv_update), METH_VARARGS, "conv_update(x, state, w, out, channels, K)."},
    {"gated_delta", reinterpret_cast<PyCFunction>(Runtime_gated_delta), METH_VARARGS,
     "gated_delta(conv, beta, alpha, ssm_a, dt_bias, state, out, key_heads, value_heads, kd, vd, eps, head_order)."},
    {"argmax", reinterpret_cast<PyCFunction>(Runtime_argmax), METH_VARARGS, "argmax(x, n) -> int (only the index leaves the GPU)."},
    {"synchronize", reinterpret_cast<PyCFunction>(Runtime_synchronize), METH_NOARGS, "cudaDeviceSynchronize."},
    {"argmax_rows", reinterpret_cast<PyCFunction>(Runtime_argmax_rows), METH_VARARGS, "argmax_rows(x, n, ntok) -> [int]."},
    {"dyn_conv", reinterpret_cast<PyCFunction>(Runtime_dyn_conv), METH_VARARGS, "dyn_conv(x, dyn, base, out, ntok, hidden, K, group, sel)."},
    {"topk_rows", reinterpret_cast<PyCFunction>(Runtime_topk_rows), METH_VARARGS, "topk_rows(x, n, ntok, k) -> (indices, values)."},
    {"copy", reinterpret_cast<PyCFunction>(Runtime_copy), METH_VARARGS, "copy(dst, dst_off, src, src_off, n)."},
    {"matvec", reinterpret_cast<PyCFunction>(Runtime_matvec), METH_VARARGS,
     "matvec(name, x_f32_bytes) -> y_f32_bytes using the quantized resident tensor."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject RuntimeType = {PyVarObject_HEAD_INIT(nullptr, 0)};

PyModuleDef Module = {PyModuleDef_HEAD_INIT, "_cuda_qwen_runtime", "Persistent CUDA runtime for GGUF weights.", -1,
                      nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__cuda_qwen_runtime(void) {
    RuntimeType.tp_name = "vinf._cuda_qwen_runtime.Runtime";
    RuntimeType.tp_basicsize = sizeof(RuntimeObject);
    RuntimeType.tp_flags = Py_TPFLAGS_DEFAULT;
    RuntimeType.tp_new = PyType_GenericNew;
    RuntimeType.tp_init = reinterpret_cast<initproc>(Runtime_init);
    RuntimeType.tp_dealloc = reinterpret_cast<destructor>(Runtime_dealloc);
    RuntimeType.tp_methods = Runtime_methods;
    if (PyType_Ready(&RuntimeType) < 0) return nullptr;
    PyObject *module = PyModule_Create(&Module);
    if (module == nullptr) return nullptr;
    Py_INCREF(&RuntimeType);
    if (PyModule_AddObject(module, "Runtime", reinterpret_cast<PyObject *>(&RuntimeType)) < 0) {
        Py_DECREF(&RuntimeType);
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}
