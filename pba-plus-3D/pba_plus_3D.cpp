#include <Python.h>
#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"

#include "pba/pba3D.h"

PyObject* pba3DVoronoiDiagram_impl(PyObject*, PyObject *args) {
	PyObject* o;
	int phase1Band, phase2Band, phase3Band;

	if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &o, &phase1Band, &phase2Band, &phase3Band))
		return NULL;

	PyArrayObject* arr = (PyArrayObject*) PyArray_FROM_OTF(o, NPY_INT, NPY_ARRAY_INOUT_ARRAY);
	if (arr == NULL)
		return NULL;

	if (PyArray_NDIM(arr) != 3 ||
		PyArray_DIM(arr, 0) != PyArray_DIM(arr, 1) ||
		PyArray_DIM(arr, 0) != PyArray_DIM(arr, 2)
	) {
		PyArray_DiscardWritebackIfCopy(arr);
		Py_DECREF(arr);
		PyErr_SetString(PyExc_RuntimeError, "input array must have dimensions n*n*n");
		return NULL;
	}

	int textureSize = PyArray_DIM(arr, 0);
	int* data = (int*) PyArray_DATA(arr);

	pba3DInitialization(textureSize);
	pba3DVoronoiDiagram(data, data, phase1Band, phase2Band, phase3Band);
	pba3DDeinitialization();

	PyArray_ResolveWritebackIfCopy(arr);
	Py_DECREF(arr);

	Py_RETURN_NONE;
}

PyObject* pba3DDistance_impl(PyObject*, PyObject* args) {
	PyObject* oIn, * oOut;
	int phase1Band, phase2Band, phase3Band;

	if (!PyArg_ParseTuple(args, "O!O!iii", &PyArray_Type, &oIn, &PyArray_Type, &oOut, &phase1Band, &phase2Band, &phase3Band))
		return NULL;

	PyArrayObject* arrIn = (PyArrayObject*)PyArray_FROM_OTF(oIn, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
	if (arrIn == NULL)
		return NULL;

	if (PyArray_NDIM(arrIn) != 3 ||
		PyArray_DIM(arrIn, 0) != PyArray_DIM(arrIn, 1) ||
		PyArray_DIM(arrIn, 0) != PyArray_DIM(arrIn, 2)
	) {
		PyArray_DiscardWritebackIfCopy(arrIn);
		Py_DECREF(arrIn);
		PyErr_SetString(PyExc_RuntimeError, "input array must have dimensions n*n*n");
		return NULL;
	}

	PyArrayObject* arrOut = (PyArrayObject*)PyArray_FROM_OTF(oOut, NPY_FLOAT, NPY_ARRAY_OUT_ARRAY);
	if (arrOut == NULL) {
		PyArray_DiscardWritebackIfCopy(arrIn);
		Py_DECREF(arrIn);
		return NULL;
	}

	if (PyArray_NDIM(arrOut) != 3 ||
		PyArray_DIM(arrOut, 0) != PyArray_DIM(arrIn, 0) ||
		PyArray_DIM(arrOut, 1) != PyArray_DIM(arrIn, 1) ||
		PyArray_DIM(arrOut, 2) != PyArray_DIM(arrIn, 2)
	) {
		PyArray_DiscardWritebackIfCopy(arrIn);
		Py_DECREF(arrIn);
		PyArray_DiscardWritebackIfCopy(arrOut);
		Py_DECREF(arrOut);
		PyErr_SetString(PyExc_RuntimeError, "output array must have same dimensions as the input array");
		return NULL;
	}

	int textureSize = PyArray_DIM(arrIn, 0);
	bool* data = (bool*)PyArray_DATA(arrIn);
	float* output = (float*)PyArray_DATA(arrOut);

	pba3DInitialization(textureSize);
	pba3DEncodeAndDistance(data, output, phase1Band, phase2Band, phase3Band);
	pba3DDeinitialization();

	PyArray_ResolveWritebackIfCopy(arrIn);
	Py_DECREF(arrIn);

	PyArray_ResolveWritebackIfCopy(arrOut);
	Py_DECREF(arrOut);

	Py_RETURN_NONE;
}

static PyMethodDef pba3d_methods[] = {
	// The first property is the name exposed to Python, fast_tanh, the second is the C++
	// function name that contains the implementation.
	{ "voronoi", (PyCFunction) pba3DVoronoiDiagram_impl, METH_VARARGS, nullptr },
	{ "distance", (PyCFunction) pba3DDistance_impl, METH_VARARGS, nullptr },

	// Terminate the array with an object containing nulls.
	{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef pba3d_module = {
	PyModuleDef_HEAD_INIT,
	"pba3d",                        // Module name to use with Python import statements
	"Parallel Banding Algorithm 3D",// Module description
	-1,
	pba3d_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_pba3d() {
	_import_array();
	PyObject *module = PyModule_Create(&pba3d_module);
	PyModule_AddIntMacro(module, MARKER);
	return module;
}