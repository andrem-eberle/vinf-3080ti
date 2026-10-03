#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "config_3080ti.cuh"

__global__ void vinf_spec_verify_softmax_kernel(
    const float *logits,
    float *probabilities,
    int num_rows,
    int vocab_size) {
    int row = blockIdx.x;
    if (row >= num_rows) return;
    const float *row_logits = logits + row * vocab_size;
    float *row_probs = probabilities + row * vocab_size;
    float max_value = -CUDART_INF_F;
    for (int idx = 0; idx < vocab_size; ++idx) {
        float value = row_logits[idx];
        if (value > max_value) max_value = value;
    }
    float denom = 0.0f;
    for (int idx = 0; idx < vocab_size; ++idx) {
        float value = expf(row_logits[idx] - max_value);
        row_probs[idx] = value;
        denom += value;
    }
    for (int idx = 0; idx < vocab_size; ++idx) {
        row_probs[idx] = row_probs[idx] / denom;
    }
}

static PyObject *raise_cuda_error(const char *context, cudaError_t err) {
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s", context, cudaGetErrorString(err));
    return NULL;
}

static bool rows_to_float_array(
    PyObject *rows_obj,
    float **out,
    int expected_rows,
    int expected_cols) {
    PyObject *rows = PySequence_Fast(rows_obj, "expected sequence of logits rows");
    if (rows == nullptr) return false;
    if (PySequence_Fast_GET_SIZE(rows) != expected_rows) {
        Py_DECREF(rows);
        PyErr_SetString(PyExc_ValueError, "row count does not match num_verify_tokens");
        return false;
    }
    float *values = new float[expected_rows * expected_cols];
    for (int row = 0; row < expected_rows; ++row) {
        PyObject *row_obj = PySequence_Fast(
            PySequence_Fast_GET_ITEM(rows, row),
            "expected logits row sequence");
        if (row_obj == nullptr) {
            delete[] values;
            Py_DECREF(rows);
            return false;
        }
        if (PySequence_Fast_GET_SIZE(row_obj) != expected_cols) {
            delete[] values;
            Py_DECREF(row_obj);
            Py_DECREF(rows);
            PyErr_SetString(PyExc_ValueError, "row width does not match vocab_size");
            return false;
        }
        for (int col = 0; col < expected_cols; ++col) {
            values[row * expected_cols + col] = static_cast<float>(
                PyFloat_AsDouble(PySequence_Fast_GET_ITEM(row_obj, col)));
            if (PyErr_Occurred()) {
                delete[] values;
                Py_DECREF(row_obj);
                Py_DECREF(rows);
                return false;
            }
        }
        Py_DECREF(row_obj);
    }
    Py_DECREF(rows);
    *out = values;
    return true;
}

static PyObject *spec_verify(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *rows_obj = nullptr;
    int num_rows = 0;
    int vocab_size = 0;
    if (!PyArg_ParseTuple(args, "Oii", &rows_obj, &num_rows, &vocab_size)) return NULL;
    if (num_rows <= 0 || vocab_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "num_verify_tokens and vocab_size must be positive");
        return NULL;
    }
    float *host_logits = nullptr;
    if (!rows_to_float_array(rows_obj, &host_logits, num_rows, vocab_size)) return NULL;
    int total = num_rows * vocab_size;
    float *host_probs = new float[total]();
    float *dlogits = nullptr;
    float *dprobs = nullptr;
    cudaError_t err = cudaMalloc(&dlogits, sizeof(float) * total);
    if (err == cudaSuccess) err = cudaMalloc(&dprobs, sizeof(float) * total);
    if (err != cudaSuccess) {
        cudaFree(dlogits);
        cudaFree(dprobs);
        delete[] host_logits;
        delete[] host_probs;
        return raise_cuda_error("cudaMalloc", err);
    }
    err = cudaMemcpy(dlogits, host_logits, sizeof(float) * total, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) {
        vinf_spec_verify_softmax_kernel<<<num_rows, 1>>>(dlogits, dprobs, num_rows, vocab_size);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        err = cudaMemcpy(host_probs, dprobs, sizeof(float) * total, cudaMemcpyDeviceToHost);
    }
    cudaFree(dlogits);
    cudaFree(dprobs);
    delete[] host_logits;
    if (err != cudaSuccess) {
        delete[] host_probs;
        return raise_cuda_error("spec_verify", err);
    }
    PyObject *rows = PyList_New(num_rows);
    for (int row = 0; row < num_rows; ++row) {
        PyObject *py_row = PyList_New(vocab_size);
        for (int col = 0; col < vocab_size; ++col) {
            PyList_SET_ITEM(
                py_row,
                col,
                PyFloat_FromDouble(host_probs[row * vocab_size + col]));
        }
        PyList_SET_ITEM(rows, row, py_row);
    }
    delete[] host_probs;
    return rows;
}

static PyMethodDef Methods[] = {
    {"spec_verify", spec_verify, METH_VARARGS, "Run CUDA speculative verifier softmax rows."},
    {NULL, NULL, 0, NULL},
};
static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_cuda_spec_verify",
    "CUDA speculative verifier extension.",
    -1,
    Methods,
};
PyMODINIT_FUNC PyInit__cuda_spec_verify(void) { return PyModule_Create(&Module); }
