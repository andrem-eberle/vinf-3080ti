#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>

static PyObject *vinf_native_version(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyUnicode_FromString("vinf-native-0.0.1");
}

static PyObject *vinf_argmax_float32(PyObject *self, PyObject *args) {
    (void)self;

    Py_buffer view;
    if (!PyArg_ParseTuple(args, "y*", &view)) {
        return NULL;
    }

    if (view.len == 0 || view.len % (Py_ssize_t)sizeof(float) != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "expected non-empty float32 bytes");
        return NULL;
    }

    Py_ssize_t count = view.len / (Py_ssize_t)sizeof(float);
    const float *values = (const float *)view.buf;
    Py_ssize_t best_idx = 0;
    float best = -FLT_MAX;

    for (Py_ssize_t i = 0; i < count; ++i) {
        float value = values[i];
        if (value > best) {
            best = value;
            best_idx = i;
        }
    }

    PyBuffer_Release(&view);
    return PyLong_FromSsize_t(best_idx);
}

static PyMethodDef VinfMethods[] = {
    {"native_version", vinf_native_version, METH_NOARGS,
     "Return the native extension version string."},
    {"argmax_float32", vinf_argmax_float32, METH_VARARGS,
     "Return the index of the maximum value in a float32 byte buffer."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef VinfModule = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Native helpers for vinf.",
    -1,
    VinfMethods,
};

PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&VinfModule); }

