#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>
#include <string.h>

#include "config_3080ti.cuh"

__global__ void vinf_upload_checksum_kernel(const unsigned char *data, unsigned long long *out, size_t n) {
    unsigned long long local = 0;
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += blockDim.x * gridDim.x) {
        local += static_cast<unsigned long long>(data[idx]);
    }
    atomicAdd(out, local);
}

static PyObject *raise_cuda_error(const char *context, cudaError_t err) {
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s", context, cudaGetErrorString(err));
    return NULL;
}

static PyObject *upload_checksum(PyObject *self, PyObject *args) {
    (void)self;
    Py_buffer input;
    if (!PyArg_ParseTuple(args, "y*", &input)) {
        return NULL;
    }
    if (input.len <= 0) {
        PyBuffer_Release(&input);
        PyErr_SetString(PyExc_ValueError, "input must not be empty");
        return NULL;
    }
    Py_ssize_t nbytes = input.len;

    unsigned char *pinned = nullptr;
    unsigned char *device = nullptr;
    unsigned long long *device_sum = nullptr;
    unsigned long long host_sum = 0;

    cudaError_t err = cudaHostAlloc(reinterpret_cast<void **>(&pinned), nbytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        PyBuffer_Release(&input);
        return raise_cuda_error("cudaHostAlloc pinned staging", err);
    }
    memcpy(pinned, input.buf, nbytes);
    PyBuffer_Release(&input);

    err = cudaMalloc(&device, nbytes);
    if (err != cudaSuccess) {
        cudaFreeHost(pinned);
        return raise_cuda_error("cudaMalloc tensor", err);
    }
    err = cudaMalloc(&device_sum, sizeof(unsigned long long));
    if (err != cudaSuccess) {
        cudaFree(device);
        cudaFreeHost(pinned);
        return raise_cuda_error("cudaMalloc checksum", err);
    }
    err = cudaMemcpy(device, pinned, nbytes, cudaMemcpyHostToDevice);
    cudaFreeHost(pinned);
    if (err != cudaSuccess) {
        cudaFree(device);
        cudaFree(device_sum);
        return raise_cuda_error("cudaMemcpy staged tensor H2D", err);
    }
    err = cudaMemset(device_sum, 0, sizeof(unsigned long long));
    if (err != cudaSuccess) {
        cudaFree(device);
        cudaFree(device_sum);
        return raise_cuda_error("cudaMemset checksum", err);
    }

    int threads = vinf::rtx_3080_ti_config::num_threads;
    int blocks = 1;
    vinf_upload_checksum_kernel<<<blocks, threads>>>(device, device_sum, static_cast<size_t>(nbytes));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(device);
        cudaFree(device_sum);
        return raise_cuda_error("checksum kernel launch", err);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(device);
        cudaFree(device_sum);
        return raise_cuda_error("cudaDeviceSynchronize", err);
    }
    err = cudaMemcpy(&host_sum, device_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(device);
    cudaFree(device_sum);
    if (err != cudaSuccess) {
        return raise_cuda_error("cudaMemcpy checksum D2H", err);
    }

    return Py_BuildValue("(nK)", nbytes, host_sum);
}

static PyMethodDef Methods[] = {
    {"upload_checksum", upload_checksum, METH_VARARGS,
     "Stage bytes through pinned host memory, upload them to CUDA device memory, and return a device checksum."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_cuda_gguf_upload",
    "CUDA GGUF tensor upload smoke extension.",
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit__cuda_gguf_upload(void) { return PyModule_Create(&Module); }
