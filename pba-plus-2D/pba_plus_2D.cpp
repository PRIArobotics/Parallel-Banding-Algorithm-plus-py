#include <Python.h>

#include "pba/pba2D.h"

PyObject* pba2DInitialization_impl(PyObject*, PyObject *args) {
	int textureSize, phase1Band;

	if (!PyArg_ParseTuple(args, "ii", &textureSize, &phase1Band))
		return NULL;

	pba2DInitialization(textureSize, phase1Band);
	return NULL;
}

PyObject* pba2DDeinitialization_impl(PyObject*) {
	pba2DDeinitialization();
	return NULL;
}

PyObject* pba2DVoronoiDiagram_impl(PyObject*, PyObject *args) {
	short *input, *output;
	int phase1Band, phase2Band, phase3Band;

	// TODO
	if (!PyArg_ParseTuple(args, "iii", &phase1Band, &phase2Band, &phase3Band))
		return NULL;

	// TODO
	// pba2DVoronoiDiagram(input, output, phase1Band, phase2Band, phase3Band);
	return NULL;
}

static PyMethodDef pba2d_methods[] = {
	// The first property is the name exposed to Python, fast_tanh, the second is the C++
	// function name that contains the implementation.
	{ "initialize", (PyCFunction)pba2DInitialization_impl, METH_VARARGS, nullptr },
	{ "deinitialize", (PyCFunction)pba2DDeinitialization_impl, METH_NOARGS, nullptr },
	{ "voronoi", (PyCFunction)pba2DVoronoiDiagram_impl, METH_VARARGS, nullptr },

	// Terminate the array with an object containing nulls.
	{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef pba2d_module = {
	PyModuleDef_HEAD_INIT,
	"pba2d",                        // Module name to use with Python import statements
	"Parallel Banding Algorithm 2D",// Module description
	-1,
	pba2d_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_pba2d() {
	return PyModule_Create(&pba2d_module);
}