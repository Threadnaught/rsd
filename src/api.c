#include <time.h>
#include <sys/time.h>

#include "arsd.h"

#include <Python.h>

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#define raise_if_not_inited() \
	if(!inited){\
		PyErr_SetString(PyExc_ValueError, "Call arsd.init() before using arsd"); \
		return NULL; \
	}

PyFunctionObject* batch_picker = NULL;
int32_t inited;
static arsd_config_t config;

int32_t get_function_argument(PyObject *object, void *address){
	if(!PyFunction_Check(object)){
		PyErr_SetString(PyExc_ValueError, "File picker must be a function");
		return 0;
	}
	PyFunctionObject** address_typed = (PyFunctionObject**)address;
	*address_typed = (PyFunctionObject*)object;
	return 1;
}

int32_t pick_batch(int32_t set_i, char** dest){
	int32_t rc = -1;
	PyObject* py_set_i = NULL;
	PyObject* py_batch_size = NULL;
	PyObject* args = NULL;


	PyObject* filenames_unchecked = NULL;
	PyObject* filenames = NULL;
	//Different pointers to the same object
	PyObject* current_filename_unchecked = NULL;
	PyUnicodeObject* current_filename = NULL;
	
	PyObject* current_filename_encoded = NULL;
	
	if(!batch_picker)
		goto cleanup;

	py_batch_size = PyLong_FromLong(config.batch_size);
	py_set_i = PyLong_FromLong(set_i);

	if(config.pass_set_i){
		args = PyTuple_New(2);
		PyTuple_SetItem(args, 1, py_set_i);
	} else {
		args = PyTuple_New(1);
	}
	PyTuple_SetItem(args, 0, py_batch_size);
	
	filenames_unchecked = PyObject_CallObject((PyObject*)batch_picker, args);
	if(PyErr_Occurred() || !filenames_unchecked)
		goto cleanup;

	if(PyArray_Check(filenames_unchecked)){
		filenames = PyArray_ToList(filenames_unchecked);
	} else {
		filenames = filenames_unchecked;
	}

	if(PyList_Check(filenames)){
		int32_t file_count = PyList_Size(filenames);
		// fprintf(stderr, "List size:%i\n", file_count);

		if (file_count != config.batch_size){
			PyErr_SetString(PyExc_ValueError, "pick_batch should return batch_size of filenames");
			goto cleanup;
		}

		for(int32_t i = 0; i < file_count; i++){
			current_filename_unchecked = PyList_GetItem(filenames, i);
			if(!PyUnicode_Check(current_filename_unchecked)){
				PyErr_SetString(PyExc_ValueError, "pick_batch should return a list of string filenames");
				goto cleanup;
			}
			current_filename = (PyUnicodeObject*)current_filename_unchecked;
			current_filename_encoded = PyUnicode_AsEncodedString(current_filename,  "UTF-8", "strict");
			
			if(PyBytes_Size(current_filename_encoded) >= max_file_len - 1){
				PyErr_SetString(PyExc_ValueError, "returned filenames must be shorter than max_file_len");
				goto cleanup;
			}
			strncpy(dest[i], PyBytes_AsString(current_filename_encoded), max_file_len-1);
			Py_DECREF(current_filename_encoded);
			current_filename_encoded = NULL;
		}
		
		rc = 0;
		goto cleanup;
	}

	PyErr_SetString(PyExc_ValueError, "pick_batch should return a list of filenames");
	
	cleanup:

	if(args) Py_DECREF(args);

	if(filenames != filenames_unchecked && filenames_unchecked) Py_DECREF(filenames_unchecked);
	if(filenames) Py_DECREF(filenames);
	
	if(current_filename_encoded) Py_DECREF(current_filename_encoded);

	return rc;
}

arsd_config_t defaults(){
	arsd_config_t ret;

	ret.samplerate_hz = 44100;
	ret.clip_len_ms = 750;
	ret.clip_len_samples = -1; //Autofilled by init_decoder
	ret.run_in_samples= 5000;

	ret.batch_size = -1;
	ret.set_count = 1;
	ret.backlog_depth = 5;
	ret.thread_count = 5;
	
	return ret;
}

int32_t validate_config(arsd_config_t cfg){
	if(cfg.batch_size >= max_batch_size){
		PyErr_SetString(PyExc_RuntimeError, "max_batch_size exceeded");
		return 0;
	}
	if(cfg.set_count > max_sets){
		PyErr_SetString(PyExc_RuntimeError, "max_sets exceeded");
		return 0;
	}
	if(cfg.backlog_depth > max_backlog){
		PyErr_SetString(PyExc_RuntimeError, "max_backlog exceeded");
		return 0;
	}
	if(cfg.thread_count > max_threads){
		PyErr_SetString(PyExc_RuntimeError, "max_threads exceeded");
		return 0;
	}
	return 1;
}

PyObject* py_arsd_init(PyObject *self, PyObject *args, PyObject *kwargs){
	config = defaults();

	if(inited){
		PyErr_SetString(PyExc_RuntimeError, "AlReAdY iNiTeD");
		Py_RETURN_NONE;
	}
	
	char* keywords[] = {
		"pick_batch",
		"batch_size",
		"set_count",
		"samplerate_hz",
		"clip_len_ms",
		"run_in_samples",
		"backlog",
		"thread_count",
		NULL
	};

	if(!PyArg_ParseTupleAndKeywords(
		args,
		kwargs,
		"O&i|iiiiii",
		keywords,
		get_function_argument, &batch_picker,
		&config.batch_size,
		&config.set_count,
		&config.samplerate_hz,
		&config.clip_len_ms,
		&config.run_in_samples,
		&config.backlog_depth,
		&config.thread_count
	)){
		return NULL;
	}

	if(!validate_config(config)){
		Py_RETURN_NONE;
	}


	int64_t batch_picker_arg_count = PyLong_AsLong(PyObject_GetAttrString(PyFunction_GetCode((PyObject*)batch_picker), "co_argcount"));
	if(batch_picker_arg_count == 1){
		config.pass_set_i = 0;
	} else if (batch_picker_arg_count == 2) {
		config.pass_set_i = 1;
	} else {
		PyErr_SetString(PyExc_RuntimeError, "batch picker should have 1 or 2 arguments");
		Py_RETURN_NONE;
	}

	if(
		(init_decoder(&config) != 0) ||
		(init_scheduler(&config) != 0)
	){
		PyErr_SetString(PyExc_RuntimeError, "arsd init failed");
		Py_RETURN_NONE;
	}

	inited = 1;

	Py_RETURN_NONE;
}

PyObject* py_draw_batch(PyObject *self, PyObject *args, PyObject *kwargs){
	raise_if_not_inited();
	int32_t set_i = 0;
	float* output = NULL;
	char* keywords[] = {
		"set_i",
		NULL
	};

	if(!PyArg_ParseTupleAndKeywords(
		args,
		kwargs,
		"|i",
		keywords,
		&set_i)
	){
		return NULL;
	}

	if(set_i >= config.set_count){
		PyErr_SetString(PyExc_RuntimeError, "set_i must be less than configured set count");
		Py_RETURN_NONE;
	}

	if(NONBLOCKING_draw_batch(set_i, &output) != 0 || output == NULL){
		//Set a generic error message if none is present
		if(!PyErr_Occurred())
			PyErr_SetString(PyExc_RuntimeError, "Failed to draw clip");
		Py_RETURN_NONE;
	}
	
	npy_intp dims[2] = {config.batch_size, config.clip_len_samples};

	PyObject* arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, output);
	PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);

	return arr;
}

PyObject* py_BLOCKING_draw_clip(PyObject *self, PyObject *args, PyObject *kwargs){
	raise_if_not_inited();
	char* filename;
	float* output = NULL;
	char* keywords[] = {
		"filename",
		NULL
	};

	if(!PyArg_ParseTupleAndKeywords(
		args,
		kwargs,
		"s",
		keywords,
		&filename)
	){
		return NULL;
	}

	output = (float*)malloc(config.clip_len_samples * sizeof(float));

	if(BLOCKING_draw_clip(filename, output) != 0){
		PyErr_SetString(PyExc_RuntimeError, "Failed to draw clip");
		Py_RETURN_NONE;
	}
	
	npy_intp dims[1] = {config.clip_len_samples};

	PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, output);
	PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);

	return arr;
}


PyMethodDef arsd_methods[] = {
	{"init",				(PyCFunction*)py_arsd_init,				METH_VARARGS | METH_KEYWORDS,	""},
	{"draw_batch",			(PyCFunction*)py_draw_batch,			METH_VARARGS | METH_KEYWORDS,	""},
	{"BLOCKING_draw_clip",	(PyCFunction*)py_BLOCKING_draw_clip,		METH_VARARGS | METH_KEYWORDS,	""},
	{NULL,					NULL,									0,								NULL}
};
PyModuleDef arsd_definition ={
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