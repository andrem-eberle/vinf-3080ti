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

// KV cache layout: [kv_head][max_seq][hd].
__global__ void kv_append_kernel(const float *k, const float *v, float *kc, float *vc, int ntok, int kv_heads,
                                 int max_seq, int hd, int pos0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntok * kv_heads * hd) return;
    const int t = i / (kv_heads * hd), h = (i / hd) % kv_heads, d = i % hd;
    const size_t dst = (static_cast<size_t>(h) * max_seq + pos0 + t) * hd + d;
    kc[dst] = k[i];
    vc[dst] = v[i];
}

// GQA attention: query token t (grid.y) attends to positions [start, seq_len0 + t) (causal) or
// [start, seq_len0 + ntok - 1) for every token (bidirectional block, DFlash drafting).
// window > 0 limits each query to its last `window` positions (sliding-window attention).
// grid = (heads, ntok); dynamic smem = (seq_len0 + ntok - 1) floats.
__global__ void attention_kernel(const float *q, const float *kc, const float *vc, float *out, int heads, int kv_heads,
                                 int hd, int max_seq, int seq_len0, float scale, int bidirectional, int window) {
    extern __shared__ float probs_all[];
    const int h = blockIdx.x, t = blockIdx.y;
    const int seq_len = bidirectional ? seq_len0 + static_cast<int>(gridDim.y) - 1 : seq_len0 + t;
    const int qpos1 = seq_len0 + t;  // query position + 1
    const int start = window > 0 && qpos1 > window ? qpos1 - window : 0;
    float *probs = probs_all - start;  // index by absolute position
    const int kvh = h / (heads / kv_heads);
    const float *qh = q + (static_cast<size_t>(t) * heads + h) * hd;
    const float *kbase = kc + static_cast<size_t>(kvh) * max_seq * hd;
    const float *vbase = vc + static_cast<size_t>(kvh) * max_seq * hd;
    float local_max = -INFINITY;
    for (int p = start + threadIdx.x; p < seq_len; p += blockDim.x) {
        const float *kp = kbase + static_cast<size_t>(p) * hd;
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qh[d] * kp[d];
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
        for (int p = start; p < seq_len; ++p) acc += probs[p] * vbase[static_cast<size_t>(p) * hd + d];
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

__global__ void attention_tiled_kernel(const float *q, const float *kc, const float *vc, float *out, int heads,
                                       int kv_heads, int hd, int max_seq, int seq_len0, int ntok, float scale,
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
    const float *kbase = kc + static_cast<size_t>(kvh) * max_seq * hd;
    const float *vbase = vc + static_cast<size_t>(kvh) * max_seq * hd;
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
            ks[r * ld + d] = in ? kbase[static_cast<size_t>(k0 + r) * hd + d] : 0.0f;
            vs[r * ld + d] = in ? vbase[static_cast<size_t>(k0 + r) * hd + d] : 0.0f;
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

// Causal depthwise conv + SiLU stepping through ntok tokens. state: [channel][K] (last K inputs).
// snap (optional): state after each token, [ntok][channel][K].
__global__ void conv_update_kernel(const float *x, float *state, const float *w, float *out, int channels, int K,
                                   int ntok, float *snap) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float *st = state + static_cast<size_t>(c) * K;
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
    size_t n = 0;
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
    if (found->second.n < min_n) {
        PyErr_Format(PyExc_ValueError, "device buffer %s has %zu floats, need %zu", name, found->second.n, min_n);
        return nullptr;
    }
    return found->second.data;
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
        err = cudaMalloc(&self->argmax_out, kMaxTokens * sizeof(int));
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
    if (!PyArg_ParseTuple(args, "sn", &name, &n)) return nullptr;
    if (n <= 0) return PyErr_Format(PyExc_ValueError, "buffer %s size must be positive", name);
    auto found = self->buffers->find(name);
    if (found != self->buffers->end()) {
        cudaFree(found->second.data);
        self->buffers->erase(found);
    }
    DeviceBuffer b;
    cudaError_t err = cudaMalloc(&b.data, n * sizeof(float));
    if (err == cudaSuccess) err = cudaMemset(b.data, 0, n * sizeof(float));
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
    float *buf = find_buffer(self, name, 0);
    if (buf == nullptr) return nullptr;
    cudaError_t err = cudaMemset(buf, 0, (*self->buffers)[name].n * sizeof(float));
    if (err != cudaSuccess) return cuda_error("cudaMemset", err);
    Py_RETURN_NONE;
}

PyObject *Runtime_buffer_bytes(RuntimeObject *self, PyObject *) {
    size_t total = 0;
    for (auto &item : *self->buffers) total += item.second.n * sizeof(float);
    return PyLong_FromSize_t(total);
}

PyObject *Runtime_write(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    Py_buffer data;
    Py_ssize_t offset = 0;
    if (!PyArg_ParseTuple(args, "sy*|n", &name, &data, &offset)) return nullptr;
    float *buf = offset >= 0 ? find_buffer(self, name, static_cast<size_t>(offset) + static_cast<size_t>(data.len) / sizeof(float)) : nullptr;
    if (buf == nullptr || data.len % sizeof(float) != 0) {
        PyBuffer_Release(&data);
        if (buf != nullptr) PyErr_SetString(PyExc_ValueError, "write data must be float32 bytes");
        else if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "negative offset");
        return nullptr;
    }
    cudaError_t err = cudaMemcpy(buf + offset, data.buf, data.len, cudaMemcpyHostToDevice);
    PyBuffer_Release(&data);
    if (err != cudaSuccess) return cuda_error("buffer write", err);
    Py_RETURN_NONE;
}

PyObject *Runtime_read(RuntimeObject *self, PyObject *args) {
    const char *name = nullptr;
    Py_ssize_t n = -1, offset = 0;
    if (!PyArg_ParseTuple(args, "s|nn", &name, &n, &offset)) return nullptr;
    float *buf = find_buffer(self, name, 0);
    if (buf == nullptr) return nullptr;
    const size_t avail = (*self->buffers)[name].n;
    if (offset < 0 || static_cast<size_t>(offset) > avail) return PyErr_Format(PyExc_ValueError, "bad read offset");
    const size_t count = n < 0 ? avail - offset : static_cast<size_t>(n);
    if (offset + count > avail) return PyErr_Format(PyExc_ValueError, "read of %zu floats exceeds buffer %s", count, name);
    PyObject *out = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(count * sizeof(float)));
    if (out == nullptr) return nullptr;
    cudaError_t err = cudaMemcpy(PyBytes_AS_STRING(out), buf + offset, count * sizeof(float), cudaMemcpyDeviceToHost);
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
    const int gemm_from = self->gemm_min_rows > 0 ? self->gemm_min_rows : kMaxTokens + 1;
    if (ntok >= gemm_from) {
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
    const float *k = find_buffer(self, kname, n);
    const float *v = k ? find_buffer(self, vname, n) : nullptr;
    float *kc = v ? find_buffer(self, kcname, cache) : nullptr;
    float *vc = kc ? find_buffer(self, vcname, cache) : nullptr;
    if (vc == nullptr) return nullptr;
    kv_append_kernel<<<blocks_for(n, 256), 256>>>(k, v, kc, vc, ntok, kv_heads, max_seq, hd, pos);
    return launch_result("kv_append");
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
    const float *q = find_buffer(self, qname, n);
    const float *kc = q ? find_buffer(self, kcname, cache) : nullptr;
    const float *vc = kc ? find_buffer(self, vcname, cache) : nullptr;
    float *o = vc ? find_buffer(self, oname, n) : nullptr;
    if (o == nullptr) return nullptr;
    const size_t smem = static_cast<size_t>(seq_len + ntok - 1) * sizeof(float);
    const bool prompt_pass = self->gemm_min_rows > 0 && ntok >= self->gemm_min_rows;
    if (ntok > kMaxTokens || smem > 32 * 1024 || prompt_pass) {
        const size_t tsmem = attn_tiled_smem(hd);
        if (hd > 1024 || tsmem > 99 * 1024) return PyErr_Format(PyExc_ValueError, "head_dim %d too large", hd);
        static size_t tiled_attr = 0;
        if (tsmem > 48 * 1024 && tsmem > tiled_attr) {
            cudaError_t err = cudaFuncSetAttribute(attention_tiled_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   static_cast<int>(tsmem));
            if (err != cudaSuccess) return cuda_error("attention smem attribute", err);
            tiled_attr = tsmem;
        }
        attention_tiled_kernel<<<dim3(heads, (ntok + kAttnTQ - 1) / kAttnTQ), hd, tsmem>>>(
            q, kc, vc, o, heads, kv_heads, hd, max_seq, seq_len, ntok, 1.0f / sqrtf(static_cast<float>(hd)),
            bidirectional, window);
        return launch_result("attention");
    }
    if (smem > 48 * 1024) {
        cudaError_t err = cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        if (err != cudaSuccess) return cuda_error("attention smem attribute", err);
    }
    attention_kernel<<<dim3(heads, ntok), 256, smem>>>(q, kc, vc, o, heads, kv_heads, hd, max_seq, seq_len,
                                                        1.0f / sqrtf(static_cast<float>(hd)), bidirectional, window);
    return launch_result("attention");
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
    conv_update_kernel<<<blocks_for(channels, 256), 256>>>(x, st, w, o, channels, K, ntok, snap);
    return launch_result("conv_update");
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
    const int threads = vd < 32 ? 32 : ((vd + 31) / 32) * 32;
    gated_delta_kernel<<<value_heads, threads, 2 * kd * sizeof(float)>>>(conv, b, a, ssm_a, dt, st, o, key_heads,
                                                                         value_heads, kd, vd, eps, head_order, ntok, snap);
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
    if (ntok < 1 || ntok > kMaxTokens) return PyErr_Format(PyExc_ValueError, "ntok must be in [1, %d]", kMaxTokens);
    const float *x = find_buffer(self, xname, static_cast<size_t>(n) * ntok);
    if (x == nullptr) return nullptr;
    argmax_kernel<<<ntok, 1024>>>(x, n, self->argmax_out);
    int host[kMaxTokens];
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
    float *d = find_buffer(self, dname, static_cast<size_t>(doff + n));
    float *s = d ? find_buffer(self, sname, static_cast<size_t>(soff + n)) : nullptr;
    if (s == nullptr) return nullptr;
    cudaError_t err = cudaMemcpyAsync(d + doff, s + soff, n * sizeof(float), cudaMemcpyDeviceToDevice, 0);
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

PyObject *Runtime_set_gemm_min_rows(RuntimeObject *self, PyObject *args) {
    int n = 0;
    if (!PyArg_ParseTuple(args, "i", &n)) return nullptr;
    if (n < 0) return PyErr_Format(PyExc_ValueError, "gemm_min_rows must be >= 0");
    self->gemm_min_rows = n;
    Py_RETURN_NONE;
}

PyMethodDef Runtime_methods[] = {
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
