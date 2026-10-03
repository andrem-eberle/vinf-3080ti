#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>
#include <math.h>

#include "config_3080ti.cuh"
#include "math_utils.cuh"

__global__ void vinf_mlp_upgate_kernel(
    const float *x,
    const float *gate_w,
    const float *up_w,
    float *hidden,
    int hidden_size,
    int intermediate_size) {
    int row = blockIdx.x;
    if (row < intermediate_size) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
            gate_sum += gate_w[row * hidden_size + col] * x[col];
            up_sum += up_w[row * hidden_size + col] * x[col];
        }
        float gate_total = vinf::block_reduce_sum(gate_sum);
        float up_total = vinf::block_reduce_sum(up_sum);
        if (threadIdx.x == 0) {
            float sig = 1.0f / (1.0f + expf(-gate_total));
            hidden[row] = gate_total * sig * up_total;
        }
    }
}

__global__ void vinf_mlp_down_kernel(
    const float *hidden,
    const float *down_w,
    float *out,
    int hidden_size,
    int intermediate_size) {
    int out_row = blockIdx.x;
    if (out_row >= hidden_size) {
        return;
    }
    float sum = 0.0f;
    for (int col = threadIdx.x; col < intermediate_size; col += blockDim.x) {
        sum += down_w[out_row * intermediate_size + col] * hidden[col];
    }
    float total = vinf::block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        out[out_row] = total;
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

static PyObject *mlp(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *x_obj=nullptr, *gate_obj=nullptr, *up_obj=nullptr, *down_obj=nullptr;
    int hidden_size=0, intermediate_size=0;
    if (!PyArg_ParseTuple(args, "OOOOii", &x_obj, &gate_obj, &up_obj, &down_obj, &hidden_size, &intermediate_size)) return NULL;
    float *x=nullptr, *gate=nullptr, *up=nullptr, *down=nullptr;
    int nx=0, ng=0, nu=0, nd=0;
    if (!sequence_to_float_array(x_obj, &x, &nx)) return NULL;
    if (!sequence_to_float_array(gate_obj, &gate, &ng)) { delete[] x; return NULL; }
    if (!sequence_to_float_array(up_obj, &up, &nu)) { delete[] x; delete[] gate; return NULL; }
    if (!sequence_to_float_array(down_obj, &down, &nd)) { delete[] x; delete[] gate; delete[] up; return NULL; }
    if (nx != hidden_size || ng != intermediate_size * hidden_size || nu != intermediate_size * hidden_size || nd != hidden_size * intermediate_size) {
        delete[] x; delete[] gate; delete[] up; delete[] down;
        PyErr_SetString(PyExc_ValueError, "MLP tensor sizes do not match dimensions");
        return NULL;
    }
    float *host_out = new float[hidden_size]();
    float *dx=nullptr, *dgate=nullptr, *dup=nullptr, *ddown=nullptr, *dhidden=nullptr, *dout=nullptr;
    cudaError_t err = cudaMalloc(&dx, sizeof(float)*hidden_size);
    if (err == cudaSuccess) err = cudaMalloc(&dgate, sizeof(float)*ng);
    if (err == cudaSuccess) err = cudaMalloc(&dup, sizeof(float)*nu);
    if (err == cudaSuccess) err = cudaMalloc(&ddown, sizeof(float)*nd);
    if (err == cudaSuccess) err = cudaMalloc(&dhidden, sizeof(float)*intermediate_size);
    if (err == cudaSuccess) err = cudaMalloc(&dout, sizeof(float)*hidden_size);
    if (err != cudaSuccess) {
        cudaFree(dx); cudaFree(dgate); cudaFree(dup); cudaFree(ddown); cudaFree(dhidden); cudaFree(dout);
        delete[] x; delete[] gate; delete[] up; delete[] down; delete[] host_out;
        return raise_cuda_error("cudaMalloc", err);
    }
    err = cudaMemcpy(dx, x, sizeof(float)*hidden_size, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dgate, gate, sizeof(float)*ng, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dup, up, sizeof(float)*nu, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(ddown, down, sizeof(float)*nd, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(dx); cudaFree(dgate); cudaFree(dup); cudaFree(ddown); cudaFree(dhidden); cudaFree(dout);
        delete[] x; delete[] gate; delete[] up; delete[] down; delete[] host_out;
        return raise_cuda_error("cudaMemcpy H2D", err);
    }
    vinf_mlp_upgate_kernel<<<intermediate_size, vinf::rtx_3080_ti_config::num_threads>>>(
        dx, dgate, dup, dhidden, hidden_size, intermediate_size);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        vinf_mlp_down_kernel<<<hidden_size, vinf::rtx_3080_ti_config::num_threads>>>(
            dhidden, ddown, dout, hidden_size, intermediate_size);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(host_out, dout, sizeof(float)*hidden_size, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dgate); cudaFree(dup); cudaFree(ddown); cudaFree(dhidden); cudaFree(dout);
    delete[] x; delete[] gate; delete[] up; delete[] down;
    if (err != cudaSuccess) {
        delete[] host_out;
        return raise_cuda_error("mlp", err);
    }
    PyObject *list = PyList_New(hidden_size);
    for (int i=0; i<hidden_size; ++i) PyList_SET_ITEM(list, i, PyFloat_FromDouble(host_out[i]));
    delete[] host_out;
    return list;
}

static PyMethodDef Methods[] = {
    {"mlp", mlp, METH_VARARGS, "Run CUDA gated MLP."},
    {NULL, NULL, 0, NULL},
};
static struct PyModuleDef Module = {PyModuleDef_HEAD_INIT, "_cuda_mlp", "CUDA MLP extension.", -1, Methods};
PyMODINIT_FUNC PyInit__cuda_mlp(void) { return PyModule_Create(&Module); }
