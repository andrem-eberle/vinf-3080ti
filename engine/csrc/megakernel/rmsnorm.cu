#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>

#include "config_3080ti.cuh"
#include "math_utils.cuh"

__global__ void vinf_rmsnorm_kernel(
    const float *input, const float *weight, float *output, int n, float eps) {
    float local_sum = 0.0f;
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float value = input[idx];
        local_sum += value * value;
    }

    float total = vinf::block_reduce_sum(local_sum);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(total / static_cast<float>(n) + eps);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        output[idx] = input[idx] * inv_rms * weight[idx];
    }
}

static PyObject *raise_cuda_error(const char *context, cudaError_t err) {
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s", context, cudaGetErrorString(err));
    return NULL;
}

static bool sequence_to_float_array(PyObject *seq, float **out, int *n) {
    PyObject *fast = PySequence_Fast(seq, "expected a sequence of floats");
    if (fast == nullptr) {
        return false;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(fast);
    if (size <= 0) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "sequence must not be empty");
        return false;
    }
    float *values = new float[size];
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
        values[i] = static_cast<float>(PyFloat_AsDouble(item));
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

static PyObject *rmsnorm(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *input_obj = nullptr;
    PyObject *weight_obj = nullptr;
    float eps = 1e-6f;
    if (!PyArg_ParseTuple(args, "OOf", &input_obj, &weight_obj, &eps)) {
        return NULL;
    }

    float *host_input = nullptr;
    float *host_weight = nullptr;
    int n_input = 0;
    int n_weight = 0;
    if (!sequence_to_float_array(input_obj, &host_input, &n_input)) {
        return NULL;
    }
    if (!sequence_to_float_array(weight_obj, &host_weight, &n_weight)) {
        delete[] host_input;
        return NULL;
    }
    if (n_input != n_weight) {
        delete[] host_input;
        delete[] host_weight;
        PyErr_SetString(PyExc_ValueError, "input and weight lengths differ");
        return NULL;
    }

    float *host_output = new float[n_input]();
    float *dev_input = nullptr;
    float *dev_weight = nullptr;
    float *dev_output = nullptr;

    cudaError_t err = cudaMalloc(&dev_input, sizeof(float) * n_input);
    if (err != cudaSuccess) {
        delete[] host_input; delete[] host_weight; delete[] host_output;
        return raise_cuda_error("cudaMalloc input", err);
    }
    err = cudaMalloc(&dev_weight, sizeof(float) * n_input);
    if (err != cudaSuccess) {
        cudaFree(dev_input);
        delete[] host_input; delete[] host_weight; delete[] host_output;
        return raise_cuda_error("cudaMalloc weight", err);
    }
    err = cudaMalloc(&dev_output, sizeof(float) * n_input);
    if (err != cudaSuccess) {
        cudaFree(dev_input); cudaFree(dev_weight);
        delete[] host_input; delete[] host_weight; delete[] host_output;
        return raise_cuda_error("cudaMalloc output", err);
    }

    err = cudaMemcpy(dev_input, host_input, sizeof(float) * n_input, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) {
        err = cudaMemcpy(dev_weight, host_weight, sizeof(float) * n_input, cudaMemcpyHostToDevice);
    }
    if (err != cudaSuccess) {
        cudaFree(dev_input); cudaFree(dev_weight); cudaFree(dev_output);
        delete[] host_input; delete[] host_weight; delete[] host_output;
        return raise_cuda_error("cudaMemcpy H2D", err);
    }

    vinf_rmsnorm_kernel<<<1, vinf::rtx_3080_ti_config::num_threads>>>(
        dev_input, dev_weight, dev_output, n_input, eps);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(host_output, dev_output, sizeof(float) * n_input, cudaMemcpyDeviceToHost);
    }

    cudaFree(dev_input); cudaFree(dev_weight); cudaFree(dev_output);
    delete[] host_input; delete[] host_weight;
    if (err != cudaSuccess) {
        delete[] host_output;
        return raise_cuda_error("rmsnorm", err);
    }

    PyObject *list = PyList_New(n_input);
    for (int i = 0; i < n_input; ++i) {
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(host_output[i]));
    }
    delete[] host_output;
    return list;
}

static PyMethodDef Methods[] = {
    {"rmsnorm", rmsnorm, METH_VARARGS, "Run CUDA RMSNorm over float lists."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_cuda_rmsnorm",
    "CUDA RMSNorm extension.",
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit__cuda_rmsnorm(void) { return PyModule_Create(&Module); }

