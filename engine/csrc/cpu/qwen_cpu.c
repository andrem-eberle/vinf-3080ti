// CPU runtime for qwen35 layers that do not fit in VRAM (Phase 31e).
//
// Weights stay in the GGUF memory map (read in place, never copied or streamed over PCIe).
// Ops mirror vinf._cuda_qwen_runtime.Runtime so the executor's layer code runs unchanged on
// either device. Activations are float32 buffers [ntok][...].
//
// Quantized matmul: activations are quantized to int8 per 32 elements (scale d, sum d*sum(q));
// each weight group of 32 is unpacked into a 32-byte vector once and dotted with up to 8
// token rows using AVX2 maddubs (u8 x s8 pairs). Rows are split across OpenMP threads.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "qwen_iq_tables.h"

enum { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ4_NL = 20, T_IQ3_S = 21, T_IQ4_XS = 23,
       T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ2_S = 22 };
#define MAX_TOK 8

static int traits(int type, int *bs, int *ts) {
    switch (type) {
        case T_F32: *bs = 1; *ts = 4; return 1;
        case T_F16: *bs = 1; *ts = 2; return 1;
        case T_Q8_0: *bs = 32; *ts = 34; return 1;
        case T_Q2_K: *bs = 256; *ts = 84; return 1;
        case T_Q3_K: *bs = 256; *ts = 110; return 1;
        case T_Q4_K: *bs = 256; *ts = 144; return 1;
        case T_Q5_K: *bs = 256; *ts = 176; return 1;
        case T_Q6_K: *bs = 256; *ts = 210; return 1;
        case T_IQ4_NL: *bs = 32; *ts = 18; return 1;
        case T_IQ3_S: *bs = 256; *ts = 110; return 1;
        case T_IQ4_XS: *bs = 256; *ts = 136; return 1;
        case T_IQ2_XXS: *bs = 256; *ts = 66; return 1;
        case T_IQ2_XS: *bs = 256; *ts = 74; return 1;
        case T_IQ2_S: *bs = 256; *ts = 82; return 1;
        case T_IQ3_XXS: *bs = 256; *ts = 98; return 1;        default: return 0;
    }
}

static const int8_t kIQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static uint32_t g_iq3s_grid[512];

static inline float f16_to_f32(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return _cvtsh_ss(h);
}

static inline void k4_scale_min(const uint8_t *s, int g, int *sc, int *m) {
    if (g < 4) {
        *sc = s[g] & 63;
        *m = s[g + 4] & 63;
    } else {
        *sc = (s[g + 4] & 0x0F) | ((s[g - 4] >> 6) << 4);
        *m = (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
    }
}

static inline __m256i dot_u8s8(__m256i w, __m256i x) {  // -> 8 x int32, lane k covers bytes 4k..4k+3
    return _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), _mm256_set1_epi16(1));
}

static inline __m256i dot_s8s8(__m256i w, __m256i x) {
    return dot_u8s8(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

static inline __m256i loadu256(const void *p) { return _mm256_loadu_si256((const __m256i *)p); }

// sign_mask64[s]: byte j = 0xFF where bit j of s is set (ggml sign bytes for 8 weights).
static uint64_t sign_mask64[256];
static void init_sign_masks(void) {
    for (int s = 0; s < 256; ++s) {
        uint64_t m = 0;
        for (int j = 0; j < 8; ++j)
            if ((s >> j) & 1) m |= (uint64_t)0xFF << (8 * j);
        sign_mask64[s] = m;
    }
}

static inline uint64_t grid64(const uint8_t *grid, int entry) {
    uint64_t v;
    memcpy(&v, grid + (size_t)entry * 8, 8);
    return v;
}

static inline uint32_t grid32(const uint8_t *grid, int entry) {
    uint32_t v;
    memcpy(&v, grid + (size_t)entry * 4, 4);
    return v;
}

static inline __m256i apply_signs(__m256i mag, __m256i m) { return _mm256_sub_epi8(_mm256_xor_si256(mag, m), m); }

static inline __m256i iq4_lookup32(const uint8_t *qs16) {
    const __m128i tab = _mm_loadu_si128((const __m128i *)kIQ4NL);
    const __m128i q = _mm_loadu_si128((const __m128i *)qs16);
    const __m128i m = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_shuffle_epi8(tab, _mm_and_si128(q, m));
    const __m128i hi = _mm_shuffle_epi8(tab, _mm_and_si128(_mm_srli_epi16(q, 4), m));
    return _mm256_set_m128i(hi, lo);
}

// Quantized activations for ntok rows: xq [ntok][cols], xd/xs [ntok][cols/32].
typedef struct {
    const int8_t *q;
    const float *d;
    const float *s;
    int cols;
} XQ;

// out[t] = W[row] . x[t] for t < ntok, one weight row (quantized types).
static void dot_row_q(int type, const uint8_t *row, int cols, const XQ *x, int ntok, float *out) {
    __m256 acc[MAX_TOK];
    float minsum[MAX_TOK];
    for (int t = 0; t < ntok; ++t) {
        acc[t] = _mm256_setzero_ps();
        minsum[t] = 0.0f;
    }
    const int ng = cols / 32;
    const __m256i m4 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(0x03), m1 = _mm256_set1_epi8(0x01);
#define FOR_T_ACC(gidx, WVEC, DOT, SCALEVEC)                                                         \
    for (int t = 0; t < ntok; ++t) {                                                                 \
        const __m256i xv = loadu256(x->q + (size_t)t * x->cols + (size_t)(gidx) * 32);               \
        const __m256 sv = _mm256_mul_ps(SCALEVEC, _mm256_set1_ps(x->d[(size_t)t * ng + (gidx)]));    \
        acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(DOT(WVEC, xv)), sv, acc[t]);                     \
    }
    switch (type) {
        case T_Q8_0:
            for (int b = 0; b < ng; ++b) {
                const uint8_t *blk = row + (size_t)b * 34;
                const __m256i w = loadu256(blk + 2);
                const __m256 sc = _mm256_set1_ps(f16_to_f32(blk));
                FOR_T_ACC(b, w, dot_s8s8, sc)
            }
            break;
        case T_IQ4_NL:
            for (int b = 0; b < ng; ++b) {
                const uint8_t *blk = row + (size_t)b * 18;
                const __m256i w = iq4_lookup32(blk + 2);
                const __m256 sc = _mm256_set1_ps(f16_to_f32(blk));
                FOR_T_ACC(b, w, dot_s8s8, sc)
            }
            break;
        case T_IQ4_XS:
            for (int b = 0; b < ng / 8; ++b) {
                const uint8_t *blk = row + (size_t)b * 136;
                const float d = f16_to_f32(blk);
                const int scales_h = blk[2] | (blk[3] << 8);
                for (int g = 0; g < 8; ++g) {
                    const int ls = ((blk[4 + g / 2] >> (4 * (g % 2))) & 0x0F) | (((scales_h >> (2 * g)) & 3) << 4);
                    const __m256i w = iq4_lookup32(blk + 8 + 16 * g);
                    const __m256 sc = _mm256_set1_ps(d * (float)(ls - 32));
                    FOR_T_ACC(b * 8 + g, w, dot_s8s8, sc)
                }
            }
            break;
        case T_Q4_K:
        case T_Q5_K:
            for (int b = 0; b < ng / 8; ++b) {
                const uint8_t *blk = row + (size_t)b * (type == T_Q4_K ? 144 : 176);
                const float d = f16_to_f32(blk), dmin = f16_to_f32(blk + 2);
                const uint8_t *qs = blk + (type == T_Q4_K ? 16 : 48);
                const __m256i qh = type == T_Q5_K ? loadu256(blk + 16) : _mm256_setzero_si256();
                for (int p = 0; p < 4; ++p) {
                    const __m256i q = loadu256(qs + 32 * p);
                    for (int half = 0; half < 2; ++half) {
                        const int g = 2 * p + half;
                        int sc, m;
                        k4_scale_min(blk + 4, g, &sc, &m);
                        __m256i w = _mm256_and_si256(half ? _mm256_srli_epi16(q, 4) : q, m4);
                        if (type == T_Q5_K)
                            w = _mm256_or_si256(w, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, g), m1), 4));
                        const __m256 scv = _mm256_set1_ps(d * (float)sc);
                        const int gi = b * 8 + g;
                        FOR_T_ACC(gi, w, dot_u8s8, scv)
                        for (int t = 0; t < ntok; ++t) minsum[t] += dmin * (float)m * x->s[(size_t)t * ng + gi];
                    }
                }
            }
            break;
        case T_Q6_K:
            for (int b = 0; b < ng / 8; ++b) {
                const uint8_t *blk = row + (size_t)b * 210;
                const float d = f16_to_f32(blk + 208);
                const int8_t *scales = (const int8_t *)(blk + 192);
                for (int g = 0; g < 8; ++g) {
                    const int chunk = g / 4, quarter = g % 4;
                    const __m256i ql = loadu256(blk + chunk * 64 + (quarter % 2) * 32);
                    const __m256i qhv = loadu256(blk + 128 + chunk * 32);
                    const __m256i lo = _mm256_and_si256(quarter >= 2 ? _mm256_srli_epi16(ql, 4) : ql, m4);
                    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qhv, 2 * quarter), m3);
                    const __m256i w = _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(hi, 4)), _mm256_set1_epi8(32));
                    const __m256 scv = _mm256_set_m128(_mm_set1_ps(d * scales[2 * g + 1]), _mm_set1_ps(d * scales[2 * g]));
                    FOR_T_ACC(b * 8 + g, w, dot_s8s8, scv)
                }
            }
            break;
        case T_Q2_K: {
            const __m256i ones = _mm256_set1_epi8(1);
            for (int b = 0; b < ng / 8; ++b) {
                const uint8_t *blk = row + (size_t)b * 84;
                const float d = f16_to_f32(blk + 80), dmin = f16_to_f32(blk + 82);
                for (int g = 0; g < 8; ++g) {
                    const __m256i qs = loadu256(blk + 16 + 32 * (g / 4));
                    const __m256i w = _mm256_and_si256(_mm256_srli_epi16(qs, 2 * (g % 4)), m3);
                    const uint8_t s0 = blk[2 * g], s1 = blk[2 * g + 1];
                    const __m256 scv = _mm256_set_m128(_mm_set1_ps(d * (s1 & 0x0F)), _mm_set1_ps(d * (s0 & 0x0F)));
                    const __m256 mnv = _mm256_set_m128(_mm_set1_ps(dmin * (s1 >> 4)), _mm_set1_ps(dmin * (s0 >> 4)));
                    const int gi = b * 8 + g;
                    for (int t = 0; t < ntok; ++t) {
                        const __m256i xv = loadu256(x->q + (size_t)t * x->cols + (size_t)gi * 32);
                        const __m256 xd = _mm256_set1_ps(x->d[(size_t)t * ng + gi]);
                        acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_u8s8(w, xv)), _mm256_mul_ps(scv, xd), acc[t]);
                        acc[t] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(dot_u8s8(ones, xv)), _mm256_mul_ps(mnv, xd), acc[t]);
                    }
                }
            }
            break;
        }
        case T_IQ2_XXS:
        case T_IQ2_XS:
        case T_IQ2_S:
        case T_IQ3_XXS: {
            // Ports of ggml dequantize_row_iq{2_xxs,2_xs,2_s,3_xxs}: 32 weights = codebook rows x signs.
            const int tsz = type == T_IQ2_XXS ? 66 : type == T_IQ2_XS ? 74 : type == T_IQ2_S ? 82 : 98;
            for (int b = 0; b < ng / 8; ++b) {
                const uint8_t *blk = row + (size_t)b * tsz;
                const float d = f16_to_f32(blk);
                for (int ib = 0; ib < 8; ++ib) {
                    uint64_t mag[4], sm[4];
                    float s0, s1;
                    if (type == T_IQ2_XXS) {
                        uint32_t aux0, aux1;
                        memcpy(&aux0, blk + 2 + 8 * ib, 4);
                        memcpy(&aux1, blk + 6 + 8 * ib, 4);
                        s0 = s1 = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
                        for (int l = 0; l < 4; ++l) {
                            mag[l] = grid64(c_iq2xxs_grid, (aux0 >> (8 * l)) & 0xFF);
                            sm[l] = sign_mask64[c_ksigns_iq2xs[(aux1 >> (7 * l)) & 127]];
                        }
                    } else if (type == T_IQ2_XS) {
                        const uint8_t sc = blk[66 + ib];
                        s0 = d * (0.5f + (float)(sc & 0x0F)) * 0.25f;
                        s1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
                        for (int l = 0; l < 4; ++l) {
                            uint16_t q;
                            memcpy(&q, blk + 2 + 2 * (4 * ib + l), 2);
                            mag[l] = grid64(c_iq2xs_grid, q & 511);
                            sm[l] = sign_mask64[c_ksigns_iq2xs[q >> 9]];
                        }
                    } else if (type == T_IQ2_S) {
                        const uint8_t sc = blk[74 + ib], qh = blk[66 + ib];
                        s0 = d * (0.5f + (float)(sc & 0x0F)) * 0.25f;
                        s1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
                        for (int l = 0; l < 4; ++l) {
                            mag[l] = grid64(c_iq2s_grid, blk[2 + 4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300));
                            sm[l] = sign_mask64[blk[34 + 4 * ib + l]];
                        }
                    } else {
                        uint32_t aux;
                        memcpy(&aux, blk + 66 + 4 * ib, 4);
                        s0 = s1 = d * (0.5f + (float)(aux >> 28)) * 0.5f;
                        for (int l = 0; l < 4; ++l) {
                            mag[l] = (uint64_t)grid32(c_iq3xxs_grid, blk[2 + 8 * ib + 2 * l]) |
                                     ((uint64_t)grid32(c_iq3xxs_grid, blk[2 + 8 * ib + 2 * l + 1]) << 32);
                            sm[l] = sign_mask64[c_ksigns_iq2xs[(aux >> (7 * l)) & 127]];
                        }
                    }
                    const __m256i wv = apply_signs(
                        _mm256_set_epi64x((long long)mag[3], (long long)mag[2], (long long)mag[1], (long long)mag[0]),
                        _mm256_set_epi64x((long long)sm[3], (long long)sm[2], (long long)sm[1], (long long)sm[0]));
                    const __m256 scv = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
                    FOR_T_ACC(b * 8 + ib, wv, dot_s8s8, scv)
                }
            }
            break;
        }
        case T_Q3_K:
            for (int b = 0; b < ng / 8; ++b) {
                const uint8_t *blk = row + (size_t)b * 110;
                const float d = f16_to_f32(blk + 108);
                const uint8_t *sb = blk + 96;
                const __m256i hm = loadu256(blk);
                for (int g = 0; g < 8; ++g) {
                    const __m256i qs = loadu256(blk + 32 + 32 * (g / 4));
                    const __m256i lo = _mm256_and_si256(_mm256_srli_epi16(qs, 2 * (g % 4)), m3);
                    const __m256i hb = _mm256_and_si256(_mm256_srli_epi16(hm, g), m1);
                    const __m256i w = _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(hb, 2)), _mm256_set1_epi8(4));
                    int scv_i[2];
                    for (int h = 0; h < 2; ++h) {
                        const int k = 2 * g + h;
                        const int l4 = (sb[k % 8] >> (4 * (k / 8))) & 0x0F;
                        const int h2 = (sb[8 + k % 4] >> (2 * (k / 4))) & 0x03;
                        scv_i[h] = (l4 | (h2 << 4)) - 32;
                    }
                    const __m256 scv = _mm256_set_m128(_mm_set1_ps(d * scv_i[1]), _mm_set1_ps(d * scv_i[0]));
                    FOR_T_ACC(b * 8 + g, w, dot_s8s8, scv)
                }
            }
            break;
        case T_IQ3_S: {
            const __m256i shuf = _mm256_set_epi8(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2,
                                                 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
            const __m256i bits = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
            for (int b = 0; b < ng / 8; ++b) {
                const uint8_t *blk = row + (size_t)b * 110;
                const float d = f16_to_f32(blk);
                for (int g = 0; g < 8; ++g) {
                    const uint8_t *qs = blk + 2 + 8 * g;
                    const int qh = blk[66 + g];
                    uint32_t e[8];
                    for (int l = 0; l < 8; ++l) e[l] = g_iq3s_grid[qs[l] | (((qh >> l) & 1) << 8)];
                    const __m256i grid = _mm256_set_epi32((int)e[7], (int)e[6], (int)e[5], (int)e[4], (int)e[3], (int)e[2], (int)e[1], (int)e[0]);
                    uint32_t sbits;
                    memcpy(&sbits, blk + 74 + 4 * g, 4);
                    const __m256i sm = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)sbits), shuf), bits), bits);
                    const __m256i w = _mm256_sub_epi8(_mm256_xor_si256(grid, sm), sm);
                    const __m256 scv = _mm256_set1_ps(d * (float)(1 + 2 * ((blk[106 + g / 2] >> (4 * (g % 2))) & 0x0F)));
                    FOR_T_ACC(b * 8 + g, w, dot_s8s8, scv)
                }
            }
            break;
        }
        default:
            break;
    }
#undef FOR_T_ACC
    for (int t = 0; t < ntok; ++t) out[t] = hsum256(acc[t]) - minsum[t];
}

// Float path (F32/F16 weights, float activations).
static void dot_row_f(int type, const uint8_t *row, int cols, const float *x, int ntok, float *out) {
    for (int t = 0; t < ntok; ++t) {
        const float *xt = x + (size_t)t * cols;
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < cols; i += 8) {
            __m256 w;
            if (type == T_F32) {
                w = _mm256_loadu_ps((const float *)row + i);
            } else {
                w = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)((const uint16_t *)row + i)));
            }
            acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(xt + i), acc);
        }
        out[t] = hsum256(acc);
    }
}

// ---- runtime object ---------------------------------------------------------------------

typedef struct {
    const uint8_t *data;
    int type, rows, cols;
    size_t row_bytes;
    Py_buffer view;
} CTensor;

typedef struct {
    float *data;
    size_t n;
} CBuffer;

typedef struct {
    PyObject_HEAD
    PyObject *tindex;
    PyObject *bindex;
    CTensor *tensors;
    int nt, ct;
    CBuffer *buffers;
    int nb, cb;
    int threads;
    int8_t *xq;
    float *xd, *xs;
    size_t xcap;
} CpuRuntime;

static CTensor *get_tensor(CpuRuntime *self, const char *name) {
    PyObject *idx = PyDict_GetItemString(self->tindex, name);
    if (idx == NULL) {
        PyErr_Format(PyExc_KeyError, "weight tensor %s is not registered on the CPU", name);
        return NULL;
    }
    return &self->tensors[PyLong_AsLong(idx)];
}

static float *get_buffer(CpuRuntime *self, const char *name, size_t min_n) {
    PyObject *idx = PyDict_GetItemString(self->bindex, name);
    if (idx == NULL) {
        PyErr_Format(PyExc_KeyError, "CPU buffer %s is not allocated", name);
        return NULL;
    }
    CBuffer *b = &self->buffers[PyLong_AsLong(idx)];
    if (b->n < min_n) {
        PyErr_Format(PyExc_ValueError, "CPU buffer %s has %zu floats, need %zu", name, b->n, min_n);
        return NULL;
    }
    return b->data;
}

static const float *get_f32(CpuRuntime *self, const char *name, size_t min_n) {
    CTensor *t = get_tensor(self, name);
    if (t == NULL) return NULL;
    if (t->type != T_F32 || (size_t)t->rows * t->cols < min_n) {
        PyErr_Format(PyExc_ValueError, "%s must be an F32 tensor with >= %zu elements", name, min_n);
        return NULL;
    }
    return (const float *)t->data;
}

static int Cpu_init(CpuRuntime *self, PyObject *args, PyObject *kwds) {
    static char *kw[] = {"iq3s_grid", "threads", NULL};
    Py_buffer grid;
    int threads = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "y*|i", kw, &grid, &threads)) return -1;
    if (grid.len != 2048) {
        PyBuffer_Release(&grid);
        PyErr_SetString(PyExc_ValueError, "iq3s_grid must be 2048 bytes");
        return -1;
    }
    memcpy(g_iq3s_grid, grid.buf, 2048);
    init_sign_masks();
    PyBuffer_Release(&grid);
    self->tindex = PyDict_New();
    self->bindex = PyDict_New();
    self->threads = threads > 0 ? threads : omp_get_num_procs() / 2;
    return 0;
}

static void Cpu_dealloc(CpuRuntime *self) {
    for (int i = 0; i < self->nt; ++i) PyBuffer_Release(&self->tensors[i].view);
    free(self->tensors);
    for (int i = 0; i < self->nb; ++i) free(self->buffers[i].data);
    free(self->buffers);
    free(self->xq);
    free(self->xd);
    free(self->xs);
    Py_XDECREF(self->tindex);
    Py_XDECREF(self->bindex);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// add_tensor(name, buffer, gguf_type, cols, rows): register weights in place (keeps the buffer alive).
static PyObject *Cpu_add_tensor(CpuRuntime *self, PyObject *args) {
    const char *name;
    PyObject *obj;
    int type, cols, rows;
    if (!PyArg_ParseTuple(args, "sOiii", &name, &obj, &type, &cols, &rows)) return NULL;
    int bs, ts;
    if (!traits(type, &bs, &ts)) return PyErr_Format(PyExc_ValueError, "unsupported GGUF type %d on CPU", type);
    if (cols <= 0 || rows <= 0 || cols % bs != 0) return PyErr_Format(PyExc_ValueError, "%s: bad shape", name);
    if (PyDict_GetItemString(self->tindex, name) != NULL) return PyErr_Format(PyExc_ValueError, "%s already registered", name);
    CTensor t;
    memset(&t, 0, sizeof(t));
    if (PyObject_GetBuffer(obj, &t.view, PyBUF_SIMPLE) != 0) return NULL;
    t.row_bytes = (size_t)(cols / bs) * ts;
    if ((size_t)t.view.len != t.row_bytes * rows) {
        PyBuffer_Release(&t.view);
        return PyErr_Format(PyExc_ValueError, "%s: buffer has %zd bytes, expected %zu", name, t.view.len, t.row_bytes * rows);
    }
    t.data = (const uint8_t *)t.view.buf;
    t.type = type;
    t.rows = rows;
    t.cols = cols;
    if (self->nt == self->ct) {
        self->ct = self->ct ? self->ct * 2 : 64;
        self->tensors = realloc(self->tensors, sizeof(CTensor) * self->ct);
    }
    self->tensors[self->nt] = t;
    PyObject *idx = PyLong_FromLong(self->nt++);
    PyDict_SetItemString(self->tindex, name, idx);
    Py_DECREF(idx);
    Py_RETURN_NONE;
}

static PyObject *Cpu_has(CpuRuntime *self, PyObject *args) {
    const char *name;
    if (!PyArg_ParseTuple(args, "s", &name)) return NULL;
    return PyBool_FromLong(PyDict_GetItemString(self->tindex, name) != NULL);
}

static PyObject *Cpu_alloc(CpuRuntime *self, PyObject *args) {
    const char *name;
    Py_ssize_t n;
    if (!PyArg_ParseTuple(args, "sn", &name, &n)) return NULL;
    if (n <= 0) return PyErr_Format(PyExc_ValueError, "buffer %s size must be positive", name);
    float *data = aligned_alloc(64, ((size_t)n * 4 + 63) / 64 * 64);
    if (data == NULL) return PyErr_NoMemory();
    memset(data, 0, (size_t)n * 4);
    PyObject *existing = PyDict_GetItemString(self->bindex, name);
    if (existing != NULL) {
        CBuffer *b = &self->buffers[PyLong_AsLong(existing)];
        free(b->data);
        b->data = data;
        b->n = (size_t)n;
        Py_RETURN_NONE;
    }
    if (self->nb == self->cb) {
        self->cb = self->cb ? self->cb * 2 : 64;
        self->buffers = realloc(self->buffers, sizeof(CBuffer) * self->cb);
    }
    self->buffers[self->nb].data = data;
    self->buffers[self->nb].n = (size_t)n;
    PyObject *idx = PyLong_FromLong(self->nb++);
    PyDict_SetItemString(self->bindex, name, idx);
    Py_DECREF(idx);
    Py_RETURN_NONE;
}

static PyObject *Cpu_zero(CpuRuntime *self, PyObject *args) {
    const char *name;
    if (!PyArg_ParseTuple(args, "s", &name)) return NULL;
    PyObject *idx = PyDict_GetItemString(self->bindex, name);
    if (idx == NULL) return PyErr_Format(PyExc_KeyError, "CPU buffer %s is not allocated", name);
    CBuffer *b = &self->buffers[PyLong_AsLong(idx)];
    memset(b->data, 0, b->n * 4);
    Py_RETURN_NONE;
}

static PyObject *Cpu_write(CpuRuntime *self, PyObject *args) {
    const char *name;
    Py_buffer data;
    Py_ssize_t offset = 0;
    if (!PyArg_ParseTuple(args, "sy*|n", &name, &data, &offset)) return NULL;
    float *b = get_buffer(self, name, (size_t)offset + (size_t)data.len / 4);
    if (b != NULL) memcpy(b + offset, data.buf, (size_t)data.len);
    PyBuffer_Release(&data);
    if (b == NULL) return NULL;
    Py_RETURN_NONE;
}

static PyObject *Cpu_read(CpuRuntime *self, PyObject *args) {
    const char *name;
    Py_ssize_t n = -1, offset = 0;
    if (!PyArg_ParseTuple(args, "s|nn", &name, &n, &offset)) return NULL;
    PyObject *idx = PyDict_GetItemString(self->bindex, name);
    if (idx == NULL) return PyErr_Format(PyExc_KeyError, "CPU buffer %s is not allocated", name);
    CBuffer *b = &self->buffers[PyLong_AsLong(idx)];
    if (n < 0) n = (Py_ssize_t)b->n - offset;
    if (offset < 0 || (size_t)(offset + n) > b->n) return PyErr_Format(PyExc_ValueError, "read outside buffer %s", name);
    return PyBytes_FromStringAndSize((const char *)(b->data + offset), n * 4);
}

static PyObject *Cpu_copy(CpuRuntime *self, PyObject *args) {
    const char *dn, *sn;
    Py_ssize_t doff, soff, n;
    if (!PyArg_ParseTuple(args, "snsnn", &dn, &doff, &sn, &soff, &n)) return NULL;
    float *d = get_buffer(self, dn, (size_t)(doff + n));
    float *s = d ? get_buffer(self, sn, (size_t)(soff + n)) : NULL;
    if (s == NULL) return NULL;
    memmove(d + doff, s + soff, (size_t)n * 4);
    Py_RETURN_NONE;
}

static int ensure_xq(CpuRuntime *self, size_t n) {
    if (n <= self->xcap) return 1;
    free(self->xq);
    free(self->xd);
    free(self->xs);
    self->xq = aligned_alloc(64, (n + 63) / 64 * 64);
    self->xd = malloc(n / 32 * sizeof(float) + 64);
    self->xs = malloc(n / 32 * sizeof(float) + 64);
    self->xcap = n;
    return self->xq && self->xd && self->xs;
}

// qmv(weight, x, y[, ntok]): y[t] = W x[t]; each weight row is read once and applied to all ntok rows
// (in groups of MAX_TOK while the row is cache-hot).
static PyObject *Cpu_qmv(CpuRuntime *self, PyObject *args) {
    const char *wn, *xn, *yn;
    int ntok = 1;
    if (!PyArg_ParseTuple(args, "sss|i", &wn, &xn, &yn, &ntok)) return NULL;
    if (ntok < 1) return PyErr_Format(PyExc_ValueError, "ntok must be positive");
    CTensor *t = get_tensor(self, wn);
    if (t == NULL) return NULL;
    if (t->cols % 32 != 0) return PyErr_Format(PyExc_ValueError, "%s: cols must be a multiple of 32", wn);
    const float *x = get_buffer(self, xn, (size_t)t->cols * ntok);
    float *y = x ? get_buffer(self, yn, (size_t)t->rows * ntok) : NULL;
    if (y == NULL) return NULL;
    const int cols = t->cols, rows = t->rows, type = t->type;
    const uint8_t *w = t->data;
    const size_t rb = t->row_bytes;
    Py_BEGIN_ALLOW_THREADS
    if (type == T_F32 || type == T_F16) {
        #pragma omp parallel for schedule(static) num_threads(self->threads)
        for (int r = 0; r < rows; ++r) {
            float out[MAX_TOK];
            for (int c = 0; c < ntok; c += MAX_TOK) {
                const int m = ntok - c < MAX_TOK ? ntok - c : MAX_TOK;
                dot_row_f(type, w + (size_t)r * rb, cols, x + (size_t)c * cols, m, out);
                for (int k = 0; k < m; ++k) y[(size_t)(c + k) * rows + r] = out[k];
            }
        }
    } else {
        const int ng = cols / 32;
        ensure_xq(self, (size_t)cols * ntok);
        int8_t *xq = self->xq;
        float *xd = self->xd, *xs = self->xs;
        for (int k = 0; k < ntok; ++k) {
            for (int g = 0; g < ng; ++g) {
                const float *v = x + (size_t)k * cols + (size_t)g * 32;
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
                const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
                int sum = 0;
                int8_t *q = xq + (size_t)k * cols + (size_t)g * 32;
                for (int i = 0; i < 32; ++i) {
                    const int qi = (int)lrintf(v[i] * id);
                    q[i] = (int8_t)qi;
                    sum += qi;
                }
                xd[(size_t)k * ng + g] = d;
                xs[(size_t)k * ng + g] = d * (float)sum;
            }
        }
        #pragma omp parallel for schedule(static) num_threads(self->threads)
        for (int r = 0; r < rows; ++r) {
            float out[MAX_TOK];
            for (int c = 0; c < ntok; c += MAX_TOK) {
                const int m = ntok - c < MAX_TOK ? ntok - c : MAX_TOK;
                const XQ xqs = {xq + (size_t)c * cols, xd + (size_t)c * ng, xs + (size_t)c * ng, cols};
                dot_row_q(type, w + (size_t)r * rb, cols, &xqs, m, out);
                for (int k = 0; k < m; ++k) y[(size_t)(c + k) * rows + r] = out[k];
            }
        }
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static inline float silu(float v) { return v / (1.0f + expf(-v)); }
static inline float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

static PyObject *Cpu_rmsnorm(CpuRuntime *self, PyObject *args) {
    const char *xn, *wn, *yn;
    int n, groups = 1;
    float eps;
    if (!PyArg_ParseTuple(args, "sssif|i", &xn, &wn, &yn, &n, &eps, &groups)) return NULL;
    const float *x = get_buffer(self, xn, (size_t)n * groups);
    float *y = x ? get_buffer(self, yn, (size_t)n * groups) : NULL;
    const float *w = y ? get_f32(self, wn, n) : NULL;
    if (w == NULL) return NULL;
    for (int g = 0; g < groups; ++g) {
        const float *xg = x + (size_t)g * n;
        float *yg = y + (size_t)g * n;
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += (double)xg[i] * xg[i];
        const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
        for (int i = 0; i < n; ++i) yg[i] = xg[i] * scale * w[i];
    }
    Py_RETURN_NONE;
}

static PyObject *Cpu_gated_rmsnorm(CpuRuntime *self, PyObject *args) {
    const char *xn, *wn, *zn, *yn;
    int n, groups;
    float eps;
    if (!PyArg_ParseTuple(args, "ssssifi", &xn, &wn, &zn, &yn, &n, &eps, &groups)) return NULL;
    const size_t tot = (size_t)n * groups;
    const float *x = get_buffer(self, xn, tot);
    const float *z = x ? get_buffer(self, zn, tot) : NULL;
    float *y = z ? get_buffer(self, yn, tot) : NULL;
    const float *w = y ? get_f32(self, wn, n) : NULL;
    if (w == NULL) return NULL;
    for (int g = 0; g < groups; ++g) {
        const size_t base = (size_t)g * n;
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += (double)x[base + i] * x[base + i];
        const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
        for (int i = 0; i < n; ++i) y[base + i] = x[base + i] * scale * w[i] * silu(z[base + i]);
    }
    Py_RETURN_NONE;
}

static PyObject *binop(CpuRuntime *self, PyObject *args, int op) {
    const char *an, *bn, *on;
    Py_ssize_t n;
    if (!PyArg_ParseTuple(args, "sssn", &an, &bn, &on, &n)) return NULL;
    const float *a = get_buffer(self, an, (size_t)n);
    const float *b = a ? get_buffer(self, bn, (size_t)n) : NULL;
    float *o = b ? get_buffer(self, on, (size_t)n) : NULL;
    if (o == NULL) return NULL;
    for (Py_ssize_t i = 0; i < n; ++i) o[i] = op == 0 ? a[i] + b[i] : op == 1 ? silu(a[i]) * b[i] : a[i] * sigm(b[i]);
    Py_RETURN_NONE;
}
static PyObject *Cpu_add(CpuRuntime *s, PyObject *a) { return binop(s, a, 0); }
static PyObject *Cpu_silu_mul(CpuRuntime *s, PyObject *a) { return binop(s, a, 1); }
static PyObject *Cpu_sigmoid_mul(CpuRuntime *s, PyObject *a) { return binop(s, a, 2); }

static PyObject *Cpu_split_gated_q(CpuRuntime *self, PyObject *args) {
    const char *rn, *qn, *gn;
    int heads, hd;
    if (!PyArg_ParseTuple(args, "sssii", &rn, &qn, &gn, &heads, &hd)) return NULL;
    const size_t n = (size_t)heads * hd;
    const float *raw = get_buffer(self, rn, 2 * n);
    float *q = raw ? get_buffer(self, qn, n) : NULL;
    float *g = q ? get_buffer(self, gn, n) : NULL;
    if (g == NULL) return NULL;
    for (int h = 0; h < heads; ++h) {
        memcpy(q + (size_t)h * hd, raw + (size_t)h * 2 * hd, (size_t)hd * 4);
        memcpy(g + (size_t)h * hd, raw + (size_t)h * 2 * hd + hd, (size_t)hd * 4);
    }
    Py_RETURN_NONE;
}

static PyObject *Cpu_rope(CpuRuntime *self, PyObject *args) {
    const char *xn;
    int heads, hd, rot, pos, ntok = 1;
    double base;
    if (!PyArg_ParseTuple(args, "siiiid|i", &xn, &heads, &hd, &rot, &pos, &base, &ntok)) return NULL;
    if (rot <= 0 || rot % 2 || rot > hd) return PyErr_Format(PyExc_ValueError, "invalid rotary dim %d", rot);
    float *x = get_buffer(self, xn, (size_t)heads * hd * ntok);
    if (x == NULL) return NULL;
    const int half = rot / 2;
    for (int t = 0; t < ntok; ++t) {
        for (int j = 0; j < half; ++j) {
            const double theta = (double)(pos + t) * pow(base, -2.0 * j / rot);
            const float c = (float)cos(theta), s = (float)sin(theta);
            for (int h = 0; h < heads; ++h) {
                float *head = x + ((size_t)t * heads + h) * hd;
                const float x0 = head[j], x1 = head[j + half];
                head[j] = x0 * c - x1 * s;
                head[j + half] = x1 * c + x0 * s;
            }
        }
    }
    Py_RETURN_NONE;
}

static PyObject *Cpu_kv_append(CpuRuntime *self, PyObject *args) {
    const char *kn, *vn, *kcn, *vcn;
    int kvh, max_seq, hd, pos, ntok = 1;
    if (!PyArg_ParseTuple(args, "ssssiiii|i", &kn, &vn, &kcn, &vcn, &kvh, &max_seq, &hd, &pos, &ntok)) return NULL;
    if (pos < 0 || pos + ntok > max_seq) return PyErr_Format(PyExc_ValueError, "positions outside KV cache");
    const float *k = get_buffer(self, kn, (size_t)kvh * hd * ntok);
    const float *v = k ? get_buffer(self, vn, (size_t)kvh * hd * ntok) : NULL;
    float *kc = v ? get_buffer(self, kcn, (size_t)kvh * hd * max_seq) : NULL;
    float *vc = kc ? get_buffer(self, vcn, (size_t)kvh * hd * max_seq) : NULL;
    if (vc == NULL) return NULL;
    for (int t = 0; t < ntok; ++t)
        for (int h = 0; h < kvh; ++h) {
            const size_t dst = ((size_t)h * max_seq + pos + t) * hd, src = ((size_t)t * kvh + h) * hd;
            memcpy(kc + dst, k + src, (size_t)hd * 4);
            memcpy(vc + dst, v + src, (size_t)hd * 4);
        }
    Py_RETURN_NONE;
}

static PyObject *Cpu_attention(CpuRuntime *self, PyObject *args) {
    const char *qn, *kcn, *vcn, *on;
    int heads, kvh, hd, max_seq, seq_len, ntok = 1;
    if (!PyArg_ParseTuple(args, "ssssiiiii|i", &qn, &kcn, &vcn, &on, &heads, &kvh, &hd, &max_seq, &seq_len, &ntok)) return NULL;
    if (kvh <= 0 || heads % kvh || seq_len <= 0 || seq_len + ntok - 1 > max_seq)
        return PyErr_Format(PyExc_ValueError, "invalid attention shape");
    const float *q = get_buffer(self, qn, (size_t)heads * hd * ntok);
    const float *kc = q ? get_buffer(self, kcn, (size_t)kvh * hd * max_seq) : NULL;
    const float *vc = kc ? get_buffer(self, vcn, (size_t)kvh * hd * max_seq) : NULL;
    float *o = vc ? get_buffer(self, on, (size_t)heads * hd * ntok) : NULL;
    if (o == NULL) return NULL;
    const float scale = 1.0f / sqrtf((float)hd);
    const int group = heads / kvh;
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for collapse(2) schedule(static) num_threads(self->threads)
    for (int t = 0; t < ntok; ++t)
        for (int h = 0; h < heads; ++h) {
            const int len = seq_len + t;
            const float *qh = q + ((size_t)t * heads + h) * hd;
            const float *kb = kc + (size_t)(h / group) * max_seq * hd, *vb = vc + (size_t)(h / group) * max_seq * hd;
            float *probs = malloc((size_t)len * 4);
            float m = -INFINITY;
            for (int p = 0; p < len; ++p) {
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) dot += qh[d] * kb[(size_t)p * hd + d];
                probs[p] = dot * scale;
                if (probs[p] > m) m = probs[p];
            }
            float den = 0.0f;
            for (int p = 0; p < len; ++p) {
                probs[p] = expf(probs[p] - m);
                den += probs[p];
            }
            float *oh = o + ((size_t)t * heads + h) * hd;
            for (int d = 0; d < hd; ++d) oh[d] = 0.0f;
            for (int p = 0; p < len; ++p)
                for (int d = 0; d < hd; ++d) oh[d] += probs[p] * vb[(size_t)p * hd + d];
            for (int d = 0; d < hd; ++d) oh[d] /= den;
            free(probs);
        }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject *Cpu_conv_update(CpuRuntime *self, PyObject *args) {
    const char *xn, *sn, *wn, *on, *snapn = NULL;
    int C, K, ntok = 1;
    if (!PyArg_ParseTuple(args, "ssssii|iz", &xn, &sn, &wn, &on, &C, &K, &ntok, &snapn)) return NULL;
    const float *x = get_buffer(self, xn, (size_t)C * ntok);
    float *st = x ? get_buffer(self, sn, (size_t)C * K) : NULL;
    float *o = st ? get_buffer(self, on, (size_t)C * ntok) : NULL;
    const float *w = o ? get_f32(self, wn, (size_t)C * K) : NULL;
    if (w == NULL) return NULL;
    float *snap = NULL;
    if (snapn != NULL && ntok > 1 && (snap = get_buffer(self, snapn, (size_t)C * K * (ntok - 1))) == NULL) return NULL;
    for (int t = 0; t < ntok; ++t)
        for (int c = 0; c < C; ++c) {
            float *s = st + (size_t)c * K;
            for (int k = 0; k < K - 1; ++k) s[k] = s[k + 1];
            s[K - 1] = x[(size_t)t * C + c];
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += s[k] * w[(size_t)c * K + k];
            o[(size_t)t * C + c] = silu(acc);
            if (snap && t < ntok - 1) memcpy(snap + ((size_t)t * C + c) * K, s, (size_t)K * 4);
        }
    Py_RETURN_NONE;
}

static PyObject *Cpu_gated_delta(CpuRuntime *self, PyObject *args) {
    const char *cn, *bn, *an, *ssm_an, *dtn, *sn, *on, *snapn = NULL;
    int kh, vh, kd, vd, order, ntok = 1;
    float eps;
    if (!PyArg_ParseTuple(args, "sssssssiiiifi|iz", &cn, &bn, &an, &ssm_an, &dtn, &sn, &on, &kh, &vh, &kd, &vd, &eps, &order,
                          &ntok, &snapn))
        return NULL;
    const size_t conv_dim = (size_t)2 * kh * kd + (size_t)vh * vd, state_n = (size_t)vh * kd * vd;
    const float *conv = get_buffer(self, cn, conv_dim * ntok);
    const float *beta = conv ? get_buffer(self, bn, (size_t)vh * ntok) : NULL;
    const float *alpha = beta ? get_buffer(self, an, (size_t)vh * ntok) : NULL;
    float *S_all = alpha ? get_buffer(self, sn, state_n) : NULL;
    float *out = S_all ? get_buffer(self, on, (size_t)vh * vd * ntok) : NULL;
    const float *ssm_a = out ? get_f32(self, ssm_an, vh) : NULL;
    const float *dt = ssm_a ? get_f32(self, dtn, vh) : NULL;
    if (dt == NULL) return NULL;
    float *snap = NULL;
    if (snapn != NULL && ntok > 1 && (snap = get_buffer(self, snapn, state_n * (ntok - 1))) == NULL) return NULL;
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static) num_threads(self->threads)
    for (int h = 0; h < vh; ++h) {
        const int k_head = order == 0 ? h / (vh / kh) : h % kh;
        float *S = S_all + (size_t)h * kd * vd;
        float *q = malloc((size_t)kd * 4), *k = malloc((size_t)kd * 4), *kv = malloc((size_t)vd * 4), *dl = malloc((size_t)vd * 4);
        for (int t = 0; t < ntok; ++t) {
            const float *row = conv + (size_t)t * conv_dim;
            const float *qs = row + (size_t)k_head * kd, *ks = row + (size_t)kh * kd + (size_t)k_head * kd;
            const float *vs = row + (size_t)2 * kh * kd + (size_t)h * vd;
            float qq = 0.0f, kk = 0.0f;
            for (int i = 0; i < kd; ++i) {
                qq += qs[i] * qs[i];
                kk += ks[i] * ks[i];
            }
            const float qscale = 1.0f / (sqrtf(qq + eps) * sqrtf((float)kd)), kn = sqrtf(kk + eps);
            for (int i = 0; i < kd; ++i) {
                q[i] = qs[i] * qscale;
                k[i] = ks[i] / kn;
            }
            const float b = sigm(beta[(size_t)t * vh + h]);
            const float xa = alpha[(size_t)t * vh + h] + dt[h];
            const float decay = expf(ssm_a[h] * (xa > 20.0f ? xa : log1pf(expf(xa))));
            for (int j = 0; j < vd; ++j) kv[j] = 0.0f;
            for (int i = 0; i < kd; ++i) {
                float *Si = S + (size_t)i * vd;
                for (int j = 0; j < vd; ++j) {
                    Si[j] *= decay;
                    kv[j] += Si[j] * k[i];
                }
            }
            for (int j = 0; j < vd; ++j) dl[j] = (vs[j] - kv[j]) * b;
            float *o = out + (size_t)t * vh * vd + (size_t)h * vd;
            for (int j = 0; j < vd; ++j) o[j] = 0.0f;
            for (int i = 0; i < kd; ++i) {
                float *Si = S + (size_t)i * vd;
                for (int j = 0; j < vd; ++j) {
                    Si[j] += k[i] * dl[j];
                    o[j] += Si[j] * q[i];
                }
            }
            if (snap && t < ntok - 1) memcpy(snap + (size_t)t * state_n + (size_t)h * kd * vd, S, (size_t)kd * vd * 4);
        }
        free(q);
        free(k);
        free(kv);
        free(dl);
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject *Cpu_buffer_bytes(CpuRuntime *self, PyObject *unused) {
    (void)unused;
    size_t total = 0;
    for (int i = 0; i < self->nb; ++i) total += self->buffers[i].n * 4;
    return PyLong_FromSize_t(total);
}

static PyObject *Cpu_tensor_bytes(CpuRuntime *self, PyObject *unused) {
    (void)unused;
    size_t total = 0;
    for (int i = 0; i < self->nt; ++i) total += self->tensors[i].row_bytes * self->tensors[i].rows;
    return PyLong_FromSize_t(total);
}

// ---- checkpoint conversion (module functions) --------------------------------------------
// dtype codes: 0 = F32, 1 = F16, 2 = BF16.

static inline float load_any(const uint8_t *p, int dtype, size_t i) {
    if (dtype == 0) {
        float f;
        memcpy(&f, p + 4 * i, 4);
        return f;
    }
    uint16_t h;
    memcpy(&h, p + 2 * i, 2);
    if (dtype == 1) return _cvtsh_ss(h);
    const uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// to_f32(buffer, dtype) -> float32 bytes
static PyObject *mod_to_f32(PyObject *self, PyObject *args) {
    (void)self;
    Py_buffer src;
    int dtype;
    if (!PyArg_ParseTuple(args, "y*i", &src, &dtype)) return NULL;
    const size_t width = dtype == 0 ? 4 : 2;
    if (dtype < 0 || dtype > 2 || src.len % width) {
        PyBuffer_Release(&src);
        return PyErr_Format(PyExc_ValueError, "bad dtype or length");
    }
    const size_t n = (size_t)src.len / width;
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(n * 4));
    if (out != NULL) {
        float *dst = (float *)PyBytes_AS_STRING(out);
        for (size_t i = 0; i < n; ++i) dst[i] = load_any(src.buf, dtype, i);
    }
    PyBuffer_Release(&src);
    return out;
}

// quantize_q8_0(buffer, dtype) -> GGUF Q8_0 blocks (per 32 values: f16 scale + 32 int8).
static PyObject *mod_quantize_q8_0(PyObject *self, PyObject *args) {
    (void)self;
    Py_buffer src;
    int dtype;
    if (!PyArg_ParseTuple(args, "y*i", &src, &dtype)) return NULL;
    const size_t width = dtype == 0 ? 4 : 2;
    if (dtype < 0 || dtype > 2 || src.len % width || ((size_t)src.len / width) % 32) {
        PyBuffer_Release(&src);
        return PyErr_Format(PyExc_ValueError, "bad dtype or length (element count must be a multiple of 32)");
    }
    const size_t n = (size_t)src.len / width, nb = n / 32;
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(nb * 34));
    if (out != NULL) {
        uint8_t *dst = (uint8_t *)PyBytes_AS_STRING(out);
        const uint8_t *s = src.buf;
        Py_BEGIN_ALLOW_THREADS
        #pragma omp parallel for schedule(static)
        for (long long b = 0; b < (long long)nb; ++b) {
            float v[32], amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                v[i] = load_any(s, dtype, (size_t)b * 32 + i);
                amax = fmaxf(amax, fabsf(v[i]));
            }
            const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
            uint8_t *blk = dst + (size_t)b * 34;
            const uint16_t dh = _cvtss_sh(d, 0);
            memcpy(blk, &dh, 2);
            for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t)(int8_t)lrintf(v[i] * id);
        }
        Py_END_ALLOW_THREADS
    }
    PyBuffer_Release(&src);
    return out;
}

// bilinear_scores(pred_codebook, succ_codebook, dtype, rank, pred_id, h_f32, cand_ids) -> [float]
// DFlash 2 candidate selector term: sum_r pred[pred_id][r] * h[r] * succ[cand][r] for each candidate,
// reading the (vocab x rank) codebooks in place.
static PyObject *mod_bilinear_scores(PyObject *self, PyObject *args) {
    (void)self;
    Py_buffer pc, sc, hb;
    int dtype, rank, pred_id;
    PyObject *cands;
    if (!PyArg_ParseTuple(args, "y*y*iiiy*O", &pc, &sc, &dtype, &rank, &pred_id, &hb, &cands)) return NULL;
    PyObject *fast = PySequence_Fast(cands, "candidates must be a sequence");
    PyObject *out = NULL;
    const size_t width = dtype == 0 ? 4 : 2;
    if (fast != NULL && hb.len == (Py_ssize_t)rank * 4 && (size_t)pc.len % (rank * width) == 0) {
        const size_t vocab = (size_t)pc.len / (rank * width);
        const float *h = (const float *)hb.buf;
        float *ph = malloc((size_t)rank * sizeof(float));
        if (pred_id >= 0 && (size_t)pred_id < vocab) {
            for (int r = 0; r < rank; ++r) ph[r] = load_any(pc.buf, dtype, (size_t)pred_id * rank + r) * h[r];
            const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
            out = PyList_New(n);
            for (Py_ssize_t i = 0; i < n; ++i) {
                const long cand = PyLong_AsLong(PySequence_Fast_GET_ITEM(fast, i));
                double acc = 0.0;
                if (cand >= 0 && (size_t)cand < vocab)
                    for (int r = 0; r < rank; ++r) acc += (double)ph[r] * load_any(sc.buf, dtype, (size_t)cand * rank + r);
                PyList_SET_ITEM(out, i, PyFloat_FromDouble(acc));
            }
        } else {
            PyErr_SetString(PyExc_ValueError, "predecessor id out of range");
        }
        free(ph);
    } else if (fast != NULL) {
        PyErr_SetString(PyExc_ValueError, "bad codebook/rank sizes");
    }
    Py_XDECREF(fast);
    PyBuffer_Release(&pc);
    PyBuffer_Release(&sc);
    PyBuffer_Release(&hb);
    return out;
}

static PyMethodDef module_methods[] = {
    {"bilinear_scores", mod_bilinear_scores, METH_VARARGS, "DFlash 2 selector pair scores."},
    {"to_f32", mod_to_f32, METH_VARARGS, "to_f32(buffer, dtype) -> f32 bytes (dtype 0=F32, 1=F16, 2=BF16)."},
    {"quantize_q8_0", mod_quantize_q8_0, METH_VARARGS, "quantize_q8_0(buffer, dtype) -> GGUF Q8_0 bytes."},
    {NULL, NULL, 0, NULL},
};

static PyMethodDef Cpu_methods[] = {
    {"add_tensor", (PyCFunction)Cpu_add_tensor, METH_VARARGS, "add_tensor(name, buffer, type, cols, rows)."},
    {"has", (PyCFunction)Cpu_has, METH_VARARGS, "has(name)."},
    {"alloc", (PyCFunction)Cpu_alloc, METH_VARARGS, "alloc(name, n)."},
    {"zero", (PyCFunction)Cpu_zero, METH_VARARGS, "zero(name)."},
    {"write", (PyCFunction)Cpu_write, METH_VARARGS, "write(name, f32_bytes[, offset])."},
    {"read", (PyCFunction)Cpu_read, METH_VARARGS, "read(name[, n, offset]) -> f32 bytes."},
    {"copy", (PyCFunction)Cpu_copy, METH_VARARGS, "copy(dst, dst_off, src, src_off, n)."},
    {"qmv", (PyCFunction)Cpu_qmv, METH_VARARGS, "qmv(weight, x, y[, ntok])."},
    {"rmsnorm", (PyCFunction)Cpu_rmsnorm, METH_VARARGS, "rmsnorm(x, w, y, n, eps[, groups])."},
    {"gated_rmsnorm", (PyCFunction)Cpu_gated_rmsnorm, METH_VARARGS, "gated_rmsnorm(x, w, z, y, n, eps, groups)."},
    {"add", (PyCFunction)Cpu_add, METH_VARARGS, "add(a, b, out, n)."},
    {"silu_mul", (PyCFunction)Cpu_silu_mul, METH_VARARGS, "silu_mul(gate, up, out, n)."},
    {"sigmoid_mul", (PyCFunction)Cpu_sigmoid_mul, METH_VARARGS, "sigmoid_mul(x, gate, out, n)."},
    {"split_gated_q", (PyCFunction)Cpu_split_gated_q, METH_VARARGS, "split_gated_q(raw, q, gate, heads, hd)."},
    {"rope", (PyCFunction)Cpu_rope, METH_VARARGS, "rope(x, heads, hd, rot, pos, base[, ntok])."},
    {"kv_append", (PyCFunction)Cpu_kv_append, METH_VARARGS, "kv_append(k, v, kc, vc, kv_heads, max_seq, hd, pos[, ntok])."},
    {"attention", (PyCFunction)Cpu_attention, METH_VARARGS, "attention(q, kc, vc, out, heads, kv_heads, hd, max_seq, seq_len[, ntok])."},
    {"conv_update", (PyCFunction)Cpu_conv_update, METH_VARARGS, "conv_update(x, state, w, out, C, K[, ntok, snap])."},
    {"gated_delta", (PyCFunction)Cpu_gated_delta, METH_VARARGS, "gated_delta(..., head_order[, ntok, snap])."},
    {"buffer_bytes", (PyCFunction)Cpu_buffer_bytes, METH_NOARGS, "Bytes of CPU buffers."},
    {"tensor_bytes", (PyCFunction)Cpu_tensor_bytes, METH_NOARGS, "Bytes of registered (mmap) weights."},
    {NULL, NULL, 0, NULL},
};

static PyTypeObject CpuRuntimeType = {PyVarObject_HEAD_INIT(NULL, 0)};
static struct PyModuleDef Module = {PyModuleDef_HEAD_INIT, "_cpu_qwen", "CPU qwen35 layer runtime.", -1, module_methods};

PyMODINIT_FUNC PyInit__cpu_qwen(void) {
    CpuRuntimeType.tp_name = "vinf._cpu_qwen.CpuRuntime";
    CpuRuntimeType.tp_basicsize = sizeof(CpuRuntime);
    CpuRuntimeType.tp_flags = Py_TPFLAGS_DEFAULT;
    CpuRuntimeType.tp_new = PyType_GenericNew;
    CpuRuntimeType.tp_init = (initproc)Cpu_init;
    CpuRuntimeType.tp_dealloc = (destructor)Cpu_dealloc;
    CpuRuntimeType.tp_methods = Cpu_methods;
    if (PyType_Ready(&CpuRuntimeType) < 0) return NULL;
    PyObject *m = PyModule_Create(&Module);
    if (m == NULL) return NULL;
    Py_INCREF(&CpuRuntimeType);
    PyModule_AddObject(m, "CpuRuntime", (PyObject *)&CpuRuntimeType);
    return m;
}
