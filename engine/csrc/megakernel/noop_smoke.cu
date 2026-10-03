#include "config_3080ti.cuh"
#include "instructions_abi.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>

extern "C" __global__ void vinf_noop_smoke_kernel(int *out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = VINF_OPCODE_NOOP;
        out[1] = vinf::rtx_3080_ti_config::cuda_arch;
        out[2] = vinf::rtx_3080_ti_config::num_threads;
    }
}

static PyObject *raise_cuda_error(const char *context, cudaError_t err) {
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s", context, cudaGetErrorString(err));
    return NULL;
}

static PyObject *run_noop_smoke(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    int host[3] = {-1, -1, -1};
    int *device = nullptr;

    cudaError_t err = cudaMalloc(&device, sizeof(host));
    if (err != cudaSuccess) {
        return raise_cuda_error("cudaMalloc", err);
    }

    vinf_noop_smoke_kernel<<<1, vinf::rtx_3080_ti_config::num_threads>>>(device);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(device);
        return raise_cuda_error("kernel launch", err);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(device);
        return raise_cuda_error("cudaDeviceSynchronize", err);
    }

    err = cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost);
    cudaFree(device);
    if (err != cudaSuccess) {
        return raise_cuda_error("cudaMemcpy", err);
    }

    return Py_BuildValue("(iii)", host[0], host[1], host[2]);
}

static PyObject *cuda_device_count(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return raise_cuda_error("cudaGetDeviceCount", err);
    }
    return PyLong_FromLong(count);
}

static PyMethodDef Methods[] = {
    {"run_noop_smoke", run_noop_smoke, METH_NOARGS,
     "Launch the RTX 3080 Ti NoOp CUDA smoke kernel and return its output."},
    {"cuda_device_count", cuda_device_count, METH_NOARGS,
     "Return cudaGetDeviceCount()."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_cuda_noop",
    "CUDA NoOp smoke extension.",
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit__cuda_noop(void) { return PyModule_Create(&Module); }
