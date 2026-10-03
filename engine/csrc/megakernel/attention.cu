#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>
#include <math.h>

#include "config_3080ti.cuh"

__global__ void vinf_attention_kernel(
    const float *query,
    const float *key_cache,
    const float *value_cache,
    float *out,
    float *scores,
    int kv_head_idx,
    int seq_len,
    int num_kv_heads,
    int max_seq,
    int head_dim,
    float scale) {
    extern __shared__ float shared[];
    float *shared_scores = shared;

    for (int pos = threadIdx.x; pos < seq_len; pos += blockDim.x) {
        int base = (kv_head_idx * max_seq + pos) * head_dim;
        float dot = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            dot += query[dim] * key_cache[base + dim];
        }
        shared_scores[pos] = dot * scale;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float max_score = shared_scores[0];
        for (int pos = 1; pos < seq_len; ++pos) {
            max_score = fmaxf(max_score, shared_scores[pos]);
        }
        float denom = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            float e = expf(shared_scores[pos] - max_score);
            shared_scores[pos] = e;
            denom += e;
        }
        for (int pos = 0; pos < seq_len; ++pos) {
            scores[pos] = shared_scores[pos] / denom;
        }
    }
    __syncthreads();

    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float total = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            int base = (kv_head_idx * max_seq + pos) * head_dim;
            total += scores[pos] * value_cache[base + dim];
        }
        out[dim] = total;
    }
}

static PyObject *raise_cuda_error(const char *context, cudaError_t err) {
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s", context, cudaGetErrorString(err));
    return NULL;
}

static bool sequence_to_float_array(PyObject *seq, float **out, int *n) {
    PyObject *fast = PySequence_Fast(seq, "expected a sequence of floats");
    if (fast == nullptr) return false;
    Py_ssize_t size = PySequence_Fast_GET_SIZE(fast);
    if (size <= 0) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "sequence must not be empty");
        return false;
    }
    float *values = new float[size];
    for (Py_ssize_t i = 0; i < size; ++i) {
        values[i] = static_cast<float>(PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast, i)));
        if (PyErr_Occurred()) {
            delete[] values;
            Py_DECREF(fast);
            return false;
        }
    }
    Py_DECREF(fast);
    *out = values;
    *n = static_cast<int>(size);
    return true;
}

static PyObject *attention(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *query_obj=nullptr, *key_obj=nullptr, *value_obj=nullptr;
    int kv_head_idx=0, seq_len=0, num_kv_heads=0, max_seq=0, head_dim=0;
    float scale=0.0f;
    if (!PyArg_ParseTuple(args, "OOOiiiiif", &query_obj, &key_obj, &value_obj,
                          &kv_head_idx, &seq_len, &num_kv_heads, &max_seq, &head_dim, &scale)) {
        return NULL;
    }
    float *host_q=nullptr, *host_k=nullptr, *host_v=nullptr;
    int n_q=0, n_k=0, n_v=0;
    if (!sequence_to_float_array(query_obj, &host_q, &n_q)) return NULL;
    if (!sequence_to_float_array(key_obj, &host_k, &n_k)) { delete[] host_q; return NULL; }
    if (!sequence_to_float_array(value_obj, &host_v, &n_v)) { delete[] host_q; delete[] host_k; return NULL; }
    int cache_n = num_kv_heads * max_seq * head_dim;
    if (n_q != head_dim || n_k != cache_n || n_v != cache_n || seq_len <= 0 || seq_len > max_seq || kv_head_idx < 0 || kv_head_idx >= num_kv_heads) {
        delete[] host_q; delete[] host_k; delete[] host_v;
        PyErr_SetString(PyExc_ValueError, "invalid attention shapes or indices");
        return NULL;
    }
    float *host_out = new float[head_dim]();
    float *dev_q=nullptr, *dev_k=nullptr, *dev_v=nullptr, *dev_out=nullptr, *dev_scores=nullptr;
    cudaError_t err = cudaMalloc(&dev_q, sizeof(float)*head_dim);
    if (err == cudaSuccess) err = cudaMalloc(&dev_k, sizeof(float)*cache_n);
    if (err == cudaSuccess) err = cudaMalloc(&dev_v, sizeof(float)*cache_n);
    if (err == cudaSuccess) err = cudaMalloc(&dev_out, sizeof(float)*head_dim);
    if (err == cudaSuccess) err = cudaMalloc(&dev_scores, sizeof(float)*seq_len);
    if (err != cudaSuccess) {
        cudaFree(dev_q); cudaFree(dev_k); cudaFree(dev_v); cudaFree(dev_out); cudaFree(dev_scores);
        delete[] host_q; delete[] host_k; delete[] host_v; delete[] host_out;
        return raise_cuda_error("cudaMalloc", err);
    }
    err = cudaMemcpy(dev_q, host_q, sizeof(float)*head_dim, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dev_k, host_k, sizeof(float)*cache_n, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dev_v, host_v, sizeof(float)*cache_n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(dev_q); cudaFree(dev_k); cudaFree(dev_v); cudaFree(dev_out); cudaFree(dev_scores);
        delete[] host_q; delete[] host_k; delete[] host_v; delete[] host_out;
        return raise_cuda_error("cudaMemcpy H2D", err);
    }
    vinf_attention_kernel<<<1, vinf::rtx_3080_ti_config::num_threads, sizeof(float)*seq_len>>>(
        dev_q, dev_k, dev_v, dev_out, dev_scores, kv_head_idx, seq_len, num_kv_heads, max_seq, head_dim, scale);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(host_out, dev_out, sizeof(float)*head_dim, cudaMemcpyDeviceToHost);
    cudaFree(dev_q); cudaFree(dev_k); cudaFree(dev_v); cudaFree(dev_out); cudaFree(dev_scores);
    delete[] host_q; delete[] host_k; delete[] host_v;
    if (err != cudaSuccess) {
        delete[] host_out;
        return raise_cuda_error("attention", err);
    }
    PyObject *list = PyList_New(head_dim);
    for (int i=0; i<head_dim; ++i) PyList_SET_ITEM(list, i, PyFloat_FromDouble(host_out[i]));
    delete[] host_out;
    return list;
}

static PyMethodDef Methods[] = {
    {"attention", attention, METH_VARARGS, "Run CUDA one-token attention."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_cuda_attention",
    "CUDA attention extension.",
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit__cuda_attention(void) { return PyModule_Create(&Module); }

