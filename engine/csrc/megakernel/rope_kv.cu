#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>

#include "config_3080ti.cuh"

__global__ void vinf_rope_kv_kernel(
    const float *input,
    const float *cos_values,
    const float *sin_values,
    float *rope_out,
    float *cache,
    int head_dim,
    int head_idx,
    int position,
    int num_heads,
    int max_seq) {
    int pair_idx = threadIdx.x;
    if (pair_idx * 2 + 1 < head_dim) {
        int i = pair_idx * 2;
        float x0 = input[i];
        float x1 = input[i + 1];
        float c = cos_values[i];
        float s = sin_values[i];
        float y0 = x0 * c - x1 * s;
        float y1 = x0 * s + x1 * c;
        rope_out[i] = y0;
        rope_out[i + 1] = y1;
        int base = (head_idx * max_seq + position) * head_dim;
        cache[base + i] = y0;
        cache[base + i + 1] = y1;
    }
}

static PyObject *raise_cuda_error(const char *context, cudaError_t err) {
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s", context, cudaGetErrorString(err));
    return NULL;
}

static bool sequence_to_float_array(PyObject *seq, float **out, int *n, bool allow_empty=false) {
    PyObject *fast = PySequence_Fast(seq, "expected a sequence of floats");
    if (fast == nullptr) return false;
    Py_ssize_t size = PySequence_Fast_GET_SIZE(fast);
    if (size <= 0 && !allow_empty) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "sequence must not be empty");
        return false;
    }
    float *values = new float[size > 0 ? size : 1];
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

static PyObject *rope_kv(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *input_obj = nullptr;
    PyObject *cos_obj = nullptr;
    PyObject *sin_obj = nullptr;
    int num_heads = 0, max_seq = 0, head_idx = 0, position = 0;
    if (!PyArg_ParseTuple(args, "OOOiiii", &input_obj, &cos_obj, &sin_obj,
                          &num_heads, &max_seq, &head_idx, &position)) {
        return NULL;
    }
    float *host_input=nullptr, *host_cos=nullptr, *host_sin=nullptr;
    int n_input=0, n_cos=0, n_sin=0;
    if (!sequence_to_float_array(input_obj, &host_input, &n_input)) return NULL;
    if (!sequence_to_float_array(cos_obj, &host_cos, &n_cos)) { delete[] host_input; return NULL; }
    if (!sequence_to_float_array(sin_obj, &host_sin, &n_sin)) { delete[] host_input; delete[] host_cos; return NULL; }
    if (n_input != n_cos || n_input != n_sin || n_input % 2 != 0) {
        delete[] host_input; delete[] host_cos; delete[] host_sin;
        PyErr_SetString(PyExc_ValueError, "input/cos/sin lengths must match and be even");
        return NULL;
    }
    if (num_heads <= 0 || max_seq <= 0 || head_idx < 0 || head_idx >= num_heads || position < 0 || position >= max_seq) {
        delete[] host_input; delete[] host_cos; delete[] host_sin;
        PyErr_SetString(PyExc_ValueError, "invalid KV layout indices");
        return NULL;
    }
    int head_dim = n_input;
    int cache_n = num_heads * max_seq * head_dim;
    float *host_rope = new float[head_dim]();
    float *host_cache = new float[cache_n]();
    float *dev_input=nullptr, *dev_cos=nullptr, *dev_sin=nullptr, *dev_rope=nullptr, *dev_cache=nullptr;
    cudaError_t err = cudaMalloc(&dev_input, sizeof(float)*head_dim);
    if (err == cudaSuccess) err = cudaMalloc(&dev_cos, sizeof(float)*head_dim);
    if (err == cudaSuccess) err = cudaMalloc(&dev_sin, sizeof(float)*head_dim);
    if (err == cudaSuccess) err = cudaMalloc(&dev_rope, sizeof(float)*head_dim);
    if (err == cudaSuccess) err = cudaMalloc(&dev_cache, sizeof(float)*cache_n);
    if (err != cudaSuccess) {
        cudaFree(dev_input); cudaFree(dev_cos); cudaFree(dev_sin); cudaFree(dev_rope); cudaFree(dev_cache);
        delete[] host_input; delete[] host_cos; delete[] host_sin; delete[] host_rope; delete[] host_cache;
        return raise_cuda_error("cudaMalloc", err);
    }
    err = cudaMemcpy(dev_input, host_input, sizeof(float)*head_dim, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dev_cos, host_cos, sizeof(float)*head_dim, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dev_sin, host_sin, sizeof(float)*head_dim, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemset(dev_cache, 0, sizeof(float)*cache_n);
    if (err != cudaSuccess) {
        cudaFree(dev_input); cudaFree(dev_cos); cudaFree(dev_sin); cudaFree(dev_rope); cudaFree(dev_cache);
        delete[] host_input; delete[] host_cos; delete[] host_sin; delete[] host_rope; delete[] host_cache;
        return raise_cuda_error("copy setup", err);
    }
    vinf_rope_kv_kernel<<<1, head_dim / 2>>>(dev_input, dev_cos, dev_sin, dev_rope, dev_cache, head_dim, head_idx, position, num_heads, max_seq);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(host_rope, dev_rope, sizeof(float)*head_dim, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(host_cache, dev_cache, sizeof(float)*cache_n, cudaMemcpyDeviceToHost);
    cudaFree(dev_input); cudaFree(dev_cos); cudaFree(dev_sin); cudaFree(dev_rope); cudaFree(dev_cache);
    delete[] host_input; delete[] host_cos; delete[] host_sin;
    if (err != cudaSuccess) {
        delete[] host_rope; delete[] host_cache;
        return raise_cuda_error("rope_kv", err);
    }
    PyObject *rope_list = PyList_New(head_dim);
    for (int i=0; i<head_dim; ++i) PyList_SET_ITEM(rope_list, i, PyFloat_FromDouble(host_rope[i]));
    PyObject *cache_list = PyList_New(cache_n);
    for (int i=0; i<cache_n; ++i) PyList_SET_ITEM(cache_list, i, PyFloat_FromDouble(host_cache[i]));
    delete[] host_rope; delete[] host_cache;
    return Py_BuildValue("(OO)", rope_list, cache_list);
}

static PyMethodDef Methods[] = {
    {"rope_kv", rope_kv, METH_VARARGS, "Run CUDA RoPE and append result to flat KV cache."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_cuda_rope_kv",
    "CUDA RoPE/KV extension.",
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit__cuda_rope_kv(void) { return PyModule_Create(&Module); }

