#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "config_3080ti.cuh"
#include "math_utils.cuh"

__global__ void vinf_math_smoke_kernel(const float *input, float *output, int n) {
    float local_sum = 0.0f;
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float value = input[idx];
        local_sum += value;
        output[idx] = value * 2.0f;
    }

    float total = vinf::block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        output[n] = total;
        half h = vinf::from_float<half>(total);
        output[n + 1] = vinf::to_float<half>(h);
    }

    if (n >= 4 && threadIdx.x == 0) {
        vinf::float4_pack pack = vinf::load4(input, 0);
        pack.x += 1.0f;
        pack.y += 1.0f;
        pack.z += 1.0f;
        pack.w += 1.0f;
        vinf::store4(output, n + 4, pack);
    }
}

static PyObject *raise_cuda_error(const char *context, cudaError_t err) {
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s", context, cudaGetErrorString(err));
    return NULL;
}

static PyObject *run_math_smoke(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *seq = nullptr;
    if (!PyArg_ParseTuple(args, "O", &seq)) {
        return NULL;
    }

    PyObject *fast = PySequence_Fast(seq, "expected a sequence of floats");
    if (fast == nullptr) {
        return NULL;
    }
    Py_ssize_t n_py = PySequence_Fast_GET_SIZE(fast);
    if (n_py <= 0) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "input must not be empty");
        return NULL;
    }
    int n = static_cast<int>(n_py);
    float *host_in = new float[n];
    for (Py_ssize_t i = 0; i < n_py; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
        host_in[i] = static_cast<float>(PyFloat_AsDouble(item));
        if (PyErr_Occurred()) {
            delete[] host_in;
            Py_DECREF(fast);
            return NULL;
        }
    }
    Py_DECREF(fast);

    int out_n = n + 8;
    float *host_out = new float[out_n]();
    float *dev_in = nullptr;
    float *dev_out = nullptr;

    cudaError_t err = cudaMalloc(&dev_in, sizeof(float) * n);
    if (err != cudaSuccess) {
        delete[] host_in;
        delete[] host_out;
        return raise_cuda_error("cudaMalloc input", err);
    }
    err = cudaMalloc(&dev_out, sizeof(float) * out_n);
    if (err != cudaSuccess) {
        cudaFree(dev_in);
        delete[] host_in;
        delete[] host_out;
        return raise_cuda_error("cudaMalloc output", err);
    }
    err = cudaMemcpy(dev_in, host_in, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(dev_in);
        cudaFree(dev_out);
        delete[] host_in;
        delete[] host_out;
        return raise_cuda_error("cudaMemcpy H2D", err);
    }
    err = cudaMemset(dev_out, 0, sizeof(float) * out_n);
    if (err != cudaSuccess) {
        cudaFree(dev_in);
        cudaFree(dev_out);
        delete[] host_in;
        delete[] host_out;
        return raise_cuda_error("cudaMemset", err);
    }

    vinf_math_smoke_kernel<<<1, vinf::rtx_3080_ti_config::num_threads>>>(dev_in, dev_out, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dev_in);
        cudaFree(dev_out);
        delete[] host_in;
        delete[] host_out;
        return raise_cuda_error("kernel launch", err);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(dev_in);
        cudaFree(dev_out);
        delete[] host_in;
        delete[] host_out;
        return raise_cuda_error("cudaDeviceSynchronize", err);
    }
    err = cudaMemcpy(host_out, dev_out, sizeof(float) * out_n, cudaMemcpyDeviceToHost);
    cudaFree(dev_in);
    cudaFree(dev_out);
    delete[] host_in;
    if (err != cudaSuccess) {
        delete[] host_out;
        return raise_cuda_error("cudaMemcpy D2H", err);
    }

    PyObject *list = PyList_New(out_n);
    for (int i = 0; i < out_n; ++i) {
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(host_out[i]));
    }
    delete[] host_out;
    return list;
}

static PyMethodDef Methods[] = {
    {"run_math_smoke", run_math_smoke, METH_VARARGS,
     "Run a CUDA math helper smoke kernel over a float sequence."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_cuda_math",
    "CUDA math utility smoke extension.",
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit__cuda_math(void) { return PyModule_Create(&Module); }

