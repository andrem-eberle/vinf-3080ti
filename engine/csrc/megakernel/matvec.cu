#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>

#include "config_3080ti.cuh"
#include "math_utils.cuh"

__global__ void vinf_matvec_kernel(
    const float *x, const float *w, float *y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float local = 0.0f;
    const float *w_row = w + row * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        local += x[col] * w_row[col];
    }
    float total = vinf::block_reduce_sum(local);
    if (threadIdx.x == 0) {
        y[row] = total;
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

static PyObject *matvec(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *x_obj = nullptr;
    PyObject *w_obj = nullptr;
    int rows = 0;
    int cols = 0;
    if (!PyArg_ParseTuple(args, "OOii", &x_obj, &w_obj, &rows, &cols)) {
        return NULL;
    }
    if (rows <= 0 || cols <= 0) {
        PyErr_SetString(PyExc_ValueError, "rows and cols must be positive");
        return NULL;
    }

    float *host_x = nullptr;
    float *host_w = nullptr;
    int n_x = 0;
    int n_w = 0;
    if (!sequence_to_float_array(x_obj, &host_x, &n_x)) return NULL;
    if (!sequence_to_float_array(w_obj, &host_w, &n_w)) {
        delete[] host_x;
        return NULL;
    }
    if (n_x != cols || n_w != rows * cols) {
        delete[] host_x; delete[] host_w;
        PyErr_SetString(PyExc_ValueError, "input or weight size does not match rows/cols");
        return NULL;
    }

    float *host_y = new float[rows]();
    float *dev_x = nullptr;
    float *dev_w = nullptr;
    float *dev_y = nullptr;
    cudaError_t err = cudaMalloc(&dev_x, sizeof(float) * cols);
    if (err == cudaSuccess) err = cudaMalloc(&dev_w, sizeof(float) * rows * cols);
    if (err == cudaSuccess) err = cudaMalloc(&dev_y, sizeof(float) * rows);
    if (err != cudaSuccess) {
        cudaFree(dev_x); cudaFree(dev_w); cudaFree(dev_y);
        delete[] host_x; delete[] host_w; delete[] host_y;
        return raise_cuda_error("cudaMalloc", err);
    }
    err = cudaMemcpy(dev_x, host_x, sizeof(float) * cols, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dev_w, host_w, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(dev_x); cudaFree(dev_w); cudaFree(dev_y);
        delete[] host_x; delete[] host_w; delete[] host_y;
        return raise_cuda_error("cudaMemcpy H2D", err);
    }

    vinf_matvec_kernel<<<rows, vinf::rtx_3080_ti_config::num_threads>>>(dev_x, dev_w, dev_y, rows, cols);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(host_y, dev_y, sizeof(float) * rows, cudaMemcpyDeviceToHost);
    cudaFree(dev_x); cudaFree(dev_w); cudaFree(dev_y);
    delete[] host_x; delete[] host_w;
    if (err != cudaSuccess) {
        delete[] host_y;
        return raise_cuda_error("matvec", err);
    }
    PyObject *list = PyList_New(rows);
    for (int i = 0; i < rows; ++i) {
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(host_y[i]));
    }
    delete[] host_y;
    return list;
}

static PyMethodDef Methods[] = {
    {"matvec", matvec, METH_VARARGS, "Run CUDA row-major matvec y = W x."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_cuda_matvec",
    "CUDA matvec extension.",
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit__cuda_matvec(void) { return PyModule_Create(&Module); }

