#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>

#include "config_3080ti.cuh"
#include "math_utils.cuh"

__global__ void vinf_final_norm_kernel(
    const float *x,
    const float *norm_w,
    float *normed,
    int hidden_size,
    float eps) {
    float local_sum = 0.0f;
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float v = x[idx];
        local_sum += v * v;
    }
    float total = vinf::block_reduce_sum(local_sum);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(total / static_cast<float>(hidden_size) + eps);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        normed[idx] = x[idx] * inv_rms * norm_w[idx];
    }
}

__global__ void vinf_lm_head_kernel(
    const float *normed,
    const float *lm_w,
    float *logits,
    int hidden_size,
    int vocab_size) {
    int row = blockIdx.x;
    if (row >= vocab_size) return;
    float local = 0.0f;
    const float *w = lm_w + row * hidden_size;
    for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        local += normed[col] * w[col];
    }
    float total = vinf::block_reduce_sum(local);
    if (threadIdx.x == 0) logits[row] = total;
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

static PyObject *lm_head(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *x_obj=nullptr, *norm_obj=nullptr, *lm_obj=nullptr;
    int hidden_size=0, vocab_size=0;
    float eps=1e-6f;
    if (!PyArg_ParseTuple(args, "OOOiif", &x_obj, &norm_obj, &lm_obj, &hidden_size, &vocab_size, &eps)) return NULL;
    float *x=nullptr, *norm=nullptr, *lm=nullptr;
    int nx=0, nn=0, nl=0;
    if (!sequence_to_float_array(x_obj, &x, &nx)) return NULL;
    if (!sequence_to_float_array(norm_obj, &norm, &nn)) { delete[] x; return NULL; }
    if (!sequence_to_float_array(lm_obj, &lm, &nl)) { delete[] x; delete[] norm; return NULL; }
    if (nx != hidden_size || nn != hidden_size || nl != hidden_size * vocab_size) {
        delete[] x; delete[] norm; delete[] lm;
        PyErr_SetString(PyExc_ValueError, "LM head tensor sizes do not match dimensions");
        return NULL;
    }
    float *host_logits = new float[vocab_size]();
    float *dx=nullptr, *dnorm=nullptr, *dlm=nullptr, *dnormed=nullptr, *dlogits=nullptr;
    cudaError_t err = cudaMalloc(&dx, sizeof(float)*hidden_size);
    if (err == cudaSuccess) err = cudaMalloc(&dnorm, sizeof(float)*hidden_size);
    if (err == cudaSuccess) err = cudaMalloc(&dlm, sizeof(float)*hidden_size*vocab_size);
    if (err == cudaSuccess) err = cudaMalloc(&dnormed, sizeof(float)*hidden_size);
    if (err == cudaSuccess) err = cudaMalloc(&dlogits, sizeof(float)*vocab_size);
    if (err != cudaSuccess) {
        cudaFree(dx); cudaFree(dnorm); cudaFree(dlm); cudaFree(dnormed); cudaFree(dlogits);
        delete[] x; delete[] norm; delete[] lm; delete[] host_logits;
        return raise_cuda_error("cudaMalloc", err);
    }
    err = cudaMemcpy(dx, x, sizeof(float)*hidden_size, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dnorm, norm, sizeof(float)*hidden_size, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dlm, lm, sizeof(float)*hidden_size*vocab_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(dx); cudaFree(dnorm); cudaFree(dlm); cudaFree(dnormed); cudaFree(dlogits);
        delete[] x; delete[] norm; delete[] lm; delete[] host_logits;
        return raise_cuda_error("cudaMemcpy H2D", err);
    }
    vinf_final_norm_kernel<<<1, vinf::rtx_3080_ti_config::num_threads>>>(dx, dnorm, dnormed, hidden_size, eps);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        vinf_lm_head_kernel<<<vocab_size, vinf::rtx_3080_ti_config::num_threads>>>(dnormed, dlm, dlogits, hidden_size, vocab_size);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(host_logits, dlogits, sizeof(float)*vocab_size, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dnorm); cudaFree(dlm); cudaFree(dnormed); cudaFree(dlogits);
    delete[] x; delete[] norm; delete[] lm;
    if (err != cudaSuccess) {
        delete[] host_logits;
        return raise_cuda_error("lm_head", err);
    }
    PyObject *list = PyList_New(vocab_size);
    for (int i=0; i<vocab_size; ++i) PyList_SET_ITEM(list, i, PyFloat_FromDouble(host_logits[i]));
    delete[] host_logits;
    return list;
}

static PyMethodDef Methods[] = {
    {"lm_head", lm_head, METH_VARARGS, "Run CUDA final RMSNorm and LM head."},
    {NULL, NULL, 0, NULL},
};
static struct PyModuleDef Module = {PyModuleDef_HEAD_INIT, "_cuda_lm_head", "CUDA LM head extension.", -1, Methods};
PyMODINIT_FUNC PyInit__cuda_lm_head(void) { return PyModule_Create(&Module); }

