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

static int get_function_argument(PyObject *object, void *address){
	if(!PyFunction_Check(object)){
		PyErr_SetString(PyExc_ValueError, "File picker must be a function");
		return 0;
	}
	PyFunctionObject** address_typed = (PyFunctionObject**)address;
	*address_typed = (PyFunctionObject*)object;
	return 1;
}


static PyFunctionObject* batch_picker = NULL;
static int inited; // TODO
static int batch_size;

static int pick_batch(int set_i, char** dest){
	if(!batch_picker)
		return -1;
	PyObject* args = PyTuple_New(2);
	PyTuple_SetItem(args, 0, PyLong_FromLong(set_i));
	PyTuple_SetItem(args, 1, PyLong_FromLong(batch_size));

	PyObject* filenames = PyObject_CallObject((PyObject*)batch_picker, args);
	if(!filenames)
		return -1;


	if(PyArray_Check(filenames)){
		filenames = PyArray_ToList(filenames);
	}

	if(PyList_Check(filenames)){
		int file_count = PyList_Size(filenames);
		fprintf(stderr, "List size:%i\n", file_count);

		if (file_count != batch_size){
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
	return 1;
}

static PyObject* py_arsd_init(PyObject *self, PyObject *args, PyObject *kwargs){
	
	int samplerate_hz=44100;
	int clip_len_ms=750;

	int run_in_samples=2000;

	static char* keywords[] = {
		"pick_batch",
		"batch_size",
		"samplerate_hz",
		"clip_len_ms",
		"run_in_samples",
		NULL
	};

	if(!PyArg_ParseTupleAndKeywords(
		args,
		kwargs,
		"O&i|iiii",
		keywords,
		get_function_argument, &batch_picker,
		&batch_size,
		&samplerate_hz,
		&clip_len_ms,
		&run_in_samples)
	){
		return NULL;
	}

	if(batch_size >= max_batch_size){
		PyErr_SetString(PyExc_RuntimeError, "max_batch_size exceeded");
		PyErr_Occurred();
	}
	//TODO: integrate grab and validate max batch size

	if(init(samplerate_hz, clip_len_ms, run_in_samples) != 0){
		PyErr_SetString(PyExc_RuntimeError, "arsd init failed");
		PyErr_Occurred();
	}

	Py_RETURN_NONE;
}



static PyObject* py_arsd_draw(PyObject *self, PyObject *args){
	float* output;
	int64_t output_size;


	char batch[max_batch_size][max_file_len];
	char* batch_ptrs[max_batch_size]; //TODO: got to be a better way to do this
	for(int i = 0; i < max_batch_size; i++) batch_ptrs[i] = batch[i];
	pick_batch(0, batch_ptrs);

	
	timer(for(int i = 0; i < batch_size; i++){
		fprintf(stderr, "file:%s\n", batch[i]);

		if(BLOCKING_draw_clip(batch[0], &output, &output_size) != 0){
			PyErr_SetString(PyExc_RuntimeError, "Could not draw clip");
			PyErr_Occurred();
		}

	}, draw_clips);


	

	npy_intp dims[1] = {output_size};

	PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, output);
	PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA); // TODO: there has been some debate over wheter this is a correct dellocator

	return arr;
}

static PyMethodDef arsd_methods[] = {
	{"init",				py_arsd_init,	METH_VARARGS | METH_KEYWORDS,	""},
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