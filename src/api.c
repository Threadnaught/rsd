#include <time.h>
#include <sys/time.h>

#include "arsd.h"

#include <Python.h>

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#define timer(instruction, short_name) {\
	struct timeval start, end; \
	int64_t diff; \
	gettimeofday(&start, NULL); \
	{instruction;} \
	gettimeofday(&end, NULL); \
	diff = ((end.tv_sec - start.tv_sec) * 1000 * 1000) + ((end.tv_usec - start.tv_usec)); \
	fprintf(stderr, "%s took %li us\n", #short_name, diff); \
}

// TODO: block function calls when not inited

PyFunctionObject* batch_picker = NULL;
int inited; // TODO
static arsd_config_t config;

int get_function_argument(PyObject *object, void *address){
	if(!PyFunction_Check(object)){
		PyErr_SetString(PyExc_ValueError, "File picker must be a function");
		return 0;
	}
	PyFunctionObject** address_typed = (PyFunctionObject**)address;
	*address_typed = (PyFunctionObject*)object;
	return 1;
}


int pick_batch(int set_i, char** dest){
	if(!batch_picker)
		return -1;
	PyObject* args = PyTuple_New(2);
	PyTuple_SetItem(args, 0, PyLong_FromLong(set_i));
	PyTuple_SetItem(args, 1, PyLong_FromLong(config.batch_size));

	PyObject* filenames = PyObject_CallObject((PyObject*)batch_picker, args);
	if(PyErr_Occurred() || !filenames)
		return -1;

	if(PyArray_Check(filenames)){
		filenames = PyArray_ToList(filenames);
	}

	if(PyList_Check(filenames)){
		int file_count = PyList_Size(filenames);
		// fprintf(stderr, "List size:%i\n", file_count);

		if (file_count != config.batch_size){
			PyErr_SetString(PyExc_ValueError, "pick_batch should return batch_size of filenames");
			return -1;
		}

		for(int i = 0; i < file_count; i++){
			PyObject* current_filename_unchecked = PyList_GetItem(filenames, i);
			if(!PyUnicode_Check(current_filename_unchecked)){
				PyErr_SetString(PyExc_ValueError, "pick_batch should return a list of string filenames");
				return -1;
			}
			PyUnicodeObject* current_filename = (PyUnicodeObject*)current_filename_unchecked;
			PyObject* current_filename_encoded = PyUnicode_AsEncodedString(current_filename,  "UTF-8", "strict");
			
			if(PyBytes_Size(current_filename_encoded) >= max_file_len - 1){
				PyErr_SetString(PyExc_ValueError, "returned filenames must be shorter than max_file_len");
				return -1;
			}
			strncpy(dest[i], PyBytes_AsString(current_filename_encoded), max_file_len-1);

			//TODO: deallocate all these damned objects
		}
		
		return 0;
	}

	PyErr_SetString(PyExc_ValueError, "pick_batch should return a list of filenames");
	return -1;
}

arsd_config_t defaults(){
	arsd_config_t ret;

	ret.samplerate_hz = 44100;
	ret.clip_len_ms = 750;
	ret.clip_len_samples = -1; //Autofilled by init_decoder
	ret.run_in_samples= 2000;

	ret.batch_size = -1;
	ret.set_count = -1;
	ret.backlog_depth = 5;
	ret.thread_count = 5;
	
	return ret;
}

int validate_config(arsd_config_t cfg){
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
	
	// TODO: assert each of these under max
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
		"O&ii|iiiii",
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

	if(
		(init_decoder(&config) != 0) ||
		(init_scheduler(&config) != 0)
	){
		PyErr_SetString(PyExc_RuntimeError, "arsd init failed");
		Py_RETURN_NONE;
	}

	Py_RETURN_NONE;
}


PyObject* py_BLOCKING_draw_batch(PyObject *self, PyObject *args, PyObject *kwargs){
	int set_i;
	float* output = (float*)malloc(config.batch_size * config.clip_len_samples * sizeof(float));
	char* keywords[] = {
		"set_i",
		NULL
	};

	if(!PyArg_ParseTupleAndKeywords(
		args,
		kwargs,
		"i",
		keywords,
		&set_i)
	){
		return NULL;
	}

	if(BLOCKING_draw_batch(set_i, output) != 0){
		PyErr_SetString(PyExc_RuntimeError, "Failed to draw clip");
		Py_RETURN_NONE;
	}
	
	npy_intp dims[2] = {config.batch_size, config.clip_len_samples};

	PyObject* arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, output);
	PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA); // TODO: there has been some debate over wheter this is a correct dellocator

	return arr;
}

PyObject* py_draw_batch(PyObject *self, PyObject *args, PyObject *kwargs){
	int set_i;
	float* output = NULL;
	char* keywords[] = {
		"set_i",
		NULL
	};

	if(!PyArg_ParseTupleAndKeywords(
		args,
		kwargs,
		"i",
		keywords,
		&set_i)
	){
		return NULL;
	}

	if(NONBLOCKING_draw_batch(set_i, &output) != 0 || output == NULL){
		PyErr_SetString(PyExc_RuntimeError, "Failed to draw clip");
		Py_RETURN_NONE;
	}
	
	npy_intp dims[2] = {config.batch_size, config.clip_len_samples};

	PyObject* arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, output);
	PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA); // TODO: there has been some debate over wheter this is a correct dellocator

	return arr;
}

PyMethodDef arsd_methods[] = {
	{"init",				(PyCFunction*)py_arsd_init,				METH_VARARGS | METH_KEYWORDS,	""},
	{"BLOCKING_draw_batch",	(PyCFunction*)py_BLOCKING_draw_batch,	METH_VARARGS | METH_KEYWORDS,	""},
	{"draw_batch",			(PyCFunction*)py_draw_batch,			METH_VARARGS | METH_KEYWORDS,	""},
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