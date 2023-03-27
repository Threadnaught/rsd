#include "arsd.h"

#include <Python.h>

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

static PyObject* py_arsd_init(PyObject *self, PyObject *args, PyObject *kwargs){
	char* path;
	int samplerate_hz=44100;
	int clip_len_ms=750;

	static char* keywords[] = {"path", "samplerate_hz", "clip_len_ms"};

	if(!PyArg_ParseTupleAndKeywords(
		args,
		kwargs,
		"s|ii",
		keywords,
		&path,
		&samplerate_hz,
		&clip_len_ms)
	){
		return NULL;
	}

	if(init(path, samplerate_hz, clip_len_ms) != 0){
		PyErr_SetString(PyExc_RuntimeError, "Attempt to re-init arsd");
		PyErr_Occurred();
	}

	Py_RETURN_NONE;
}

static PyObject* py_arsd_draw(PyObject *self){
	float* output;
	int output_size;
	if(BLOCKING_draw_clip(&output, &output_size) != 0){
		PyErr_SetString(PyExc_RuntimeError, "Could not draw clip");
		PyErr_Occurred();
	}
	Py_RETURN_NONE;
}

static PyMethodDef arsd_methods[] = {
	{"init",					py_arsd_init,	METH_VARARGS | METH_KEYWORDS,	""},
	{"BLOCKING_draw_clip",	py_arsd_draw,	METH_NOARGS, 					""},
	{NULL,						NULL,		0,	NULL}
};
static PyModuleDef arsd_definition ={
	PyModuleDef_HEAD_INIT,
	"arsd",
	"Audio Repetitive Sampling Decoder",
	-1,
	arsd_methods
};

PyMODINIT_FUNC PyInit_arsd(void){
	PyObject* module;

	srand(time(NULL));
	// srand(0);
	
	Py_Initialize();
	import_array();
	module = PyModule_Create(&arsd_definition);
	return module;
}