#include <time.h>
#include <sys/time.h>

#include "rsd.h"

#include <Python.h>

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#define raise_if_not_inited() \
	if(!inited){\
		PyErr_SetString(PyExc_ValueError, "Call rsd.init() before using rsd"); \
		return NULL; \
	}

PyFunctionObject* batch_picker = NULL;
int32_t inited;
static rsd_config_t config;

// RNG state for main non-worker thread
uint32_t blocking_rng_state;

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

rsd_config_t defaults(){
	rsd_config_t ret;

	ret.samplerate_hz = 44100;
	ret.clip_len_samples = 33075;
	ret.run_in_samples = 5000;

	ret.batch_size = -1;
	ret.set_count = 1;
	ret.backlog_depth = 5;
	ret.thread_count = 5;
	ret.rng_seed = -1;
	ret.verbose = 0;
	
	return ret;
}

int32_t validate_config(rsd_config_t cfg){
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

PyObject* py_rsd_init(PyObject *self, PyObject *args, PyObject *kwargs){
	if(inited){
		PyErr_SetString(PyExc_RuntimeError, "AlReAdY iNiTeD");
		Py_RETURN_NONE;
	}

	config = defaults();
	
	char* keywords[] = {
		"pick_batch",
		"batch_size",
		"set_count",
		"samplerate_hz",
		"clip_len_samples",
		"run_in_samples",
		"backlog",
		"thread_count",
		"rng_seed",
		"verbose",
		NULL
	};

	if(!PyArg_ParseTupleAndKeywords(
		args,
		kwargs,
		"O&i|iiiiiiib",
		keywords,
		get_function_argument, &batch_picker,
		&config.batch_size,
		&config.set_count,
		&config.samplerate_hz,
		&config.clip_len_samples,
		&config.run_in_samples,
		&config.backlog_depth,
		&config.thread_count,
		&config.rng_seed,
		&config.verbose
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

	blocking_rng_state = config.rng_seed;
	if(blocking_rng_state == -1) {
		blocking_rng_state = time(NULL);
	} else {
		fprintf(stderr, "WARNING: rng seed has been pinned to %i\n", blocking_rng_state);
	}

	if(
		(init_decoder(&config) != 0) ||
		(init_scheduler(&config, &blocking_rng_state) != 0)
	){
		PyErr_SetString(PyExc_RuntimeError, "rsd init failed");
		Py_RETURN_NONE;
	}

	inited = 1;

	Py_RETURN_NONE;
}

PyObject* py_draw_batch(PyObject *self, PyObject *args, PyObject *kwargs){
	raise_if_not_inited();
	int32_t set_i = 0;
	float* output_samples = NULL;
	char* output_filenames = NULL;
	char* output_filename_ptrs[max_batch_size];

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

	output_filenames = (char*)malloc(max_batch_size * max_file_len);

	for(int32_t i = 0; i < max_batch_size; i++)
		output_filename_ptrs[i] = output_filenames + (i * max_file_len);

	if(NONBLOCKING_draw_batch(set_i, &output_samples, output_filename_ptrs) != 0 || output_samples == NULL){
		free(output_filenames);
		//Set a generic error message if none is present
		if(!PyErr_Occurred())
			PyErr_SetString(PyExc_RuntimeError, "Failed to draw clip");
		Py_RETURN_NONE;
	}

	npy_intp sample_dims[2] = {config.batch_size, config.clip_len_samples};
	PyObject* sample_arr = PyArray_SimpleNewFromData(2, sample_dims, NPY_FLOAT32, output_samples);
	PyArray_ENABLEFLAGS((PyArrayObject*)sample_arr, NPY_ARRAY_OWNDATA);

	PyArray_Descr* filename_desc = PyArray_DescrFromType(NPY_STRING);
	filename_desc->elsize = max_file_len;
	npy_intp filename_dims[2] = {config.batch_size};
	PyObject* filename_arr = PyArray_NewFromDescr(&PyArray_Type, filename_desc, 1, filename_dims, NULL, output_filenames, 0, NULL);
	PyArray_ENABLEFLAGS((PyArrayObject*)filename_arr, NPY_ARRAY_OWNDATA);

	PyTupleObject* ret = PyTuple_New(2);
	PyTuple_SetItem(ret, 0, sample_arr);
	PyTuple_SetItem(ret, 1, filename_arr);
	
	return ret;
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

	int64_t seek_point_samples;
	int64_t output_samples;

	if(BLOCKING_draw_clip(filename, output, &blocking_rng_state, &seek_point_samples, &output_samples) != 0){
		PyErr_SetString(PyExc_RuntimeError, "Failed to draw clip");
		Py_RETURN_NONE;
	}
	
	npy_intp dims[1] = {config.clip_len_samples};

	PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, output);
	PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);

	return arr;
}


PyMethodDef rsd_methods[] = {
	{"init",				(PyCFunction*)py_rsd_init,				METH_VARARGS | METH_KEYWORDS,	""},
	{"draw_batch",			(PyCFunction*)py_draw_batch,			METH_VARARGS | METH_KEYWORDS,	""},
	{"BLOCKING_draw_clip",	(PyCFunction*)py_BLOCKING_draw_clip,		METH_VARARGS | METH_KEYWORDS,	""},
	{NULL,					NULL,									0,								NULL}
};
PyModuleDef rsd_definition ={
	PyModuleDef_HEAD_INIT,
	"rsd",
	"Audio Repetitive Sampling Decoder",
	-1,
	rsd_methods
};

PyMODINIT_FUNC PyInit_rsd(void){
	PyObject* module;

	Py_Initialize();
	import_array();
	module = PyModule_Create(&rsd_definition);
	return module;
}