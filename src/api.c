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

// Big, messy, ugly function to convert [['a', 'b'], ['c', 'd']] into a char*** 
// extra level of pointer (char****) is so it can be set by us in the calling scope
// output will look like [["a", "b", NULL], ["c", "d", NULL], NULL]
static int convert_file_list(PyObject *object, void *address){
	if(object == NULL){
		fprintf(stderr, "Attempting cleanup\n");
		goto cleanup;
	}

	char**** file_lists_dest = (char****) address; // I think I lost my way somewhere
	*file_lists_dest = NULL;
	if(!PyList_Check(object)){
		PyErr_SetString(PyExc_RuntimeError, "Path argument must be list of list of filepaths"); // TODO: there's got to be a better error for this
		goto cleanup;
	}
	PyListObject* all_file_lists = (PyListObject *)object;
	
	int file_list_count = PyList_Size(all_file_lists);
	fprintf(stderr, "All size:%i\n", file_list_count);
	
	char*** file_lists = (char***)malloc((file_list_count+1) * sizeof(char**)); // N+1 for NULL sentinel
	// file_lists[file_list_count] = NULL; //NULL terminate file_lists
	for(int i = 0; i <= file_list_count; i++) file_lists[i] = NULL; //populate with NULL before value is set so we know not to cleanup
	*file_lists_dest = file_lists;
	
	for(int i = 0; i < file_list_count; i++){
		PyObject* current_list_unchecked = PyList_GetItem(all_file_lists, i);
		if(!PyList_Check(current_list_unchecked)){
			PyErr_SetString(PyExc_RuntimeError, "Path argument must be list of list of filepaths"); // TODO: there's got to be a better error for this
			goto cleanup;
		}
		PyListObject* current_list = (PyListObject*)current_list_unchecked;
		int file_count = PyList_Size(current_list);
		// fprintf(stderr, "current size:%i, i:%i\n", file_count, i);

		char** file_list = (char**)malloc((file_count+1) * sizeof(char**));
		for(int i = 0; i <= file_count; i++) file_list[i] = NULL;

		file_lists[i] = file_list;


		for(int j = 0; j < file_count; j++){
			PyObject* current_file_unchecked = PyList_GetItem(current_list, j);
			if(!PyUnicode_Check(current_file_unchecked)){
				PyErr_SetString(PyExc_RuntimeError, "Path argument must be list of list of filepaths"); // TODO: there's got to be a better error for this
				goto cleanup;
			}
			file_list[j] = PyBytes_AS_STRING(PyUnicode_AsEncodedString(current_file_unchecked,  "UTF-8", "strict"));
		}
	}

	return 1;
	cleanup:
	//TODO low priority - you should only be calling init once so the mem leak ain't too bad
	return 0;
}
static PyObject* py_arsd_init(PyObject *self, PyObject *args, PyObject *kwargs){
	// char* path;
	char*** file_list = NULL;
	int samplerate_hz=44100;
	int clip_len_ms=750;
	// int run_in_samples=2000; // TODO: parameterise

	static char* keywords[] = {"path", "samplerate_hz", "clip_len_ms"};

	if(!PyArg_ParseTupleAndKeywords(
		args,
		kwargs,
		"O&|ii",
		keywords,
		convert_file_list, &file_list,
		&samplerate_hz,
		&clip_len_ms)
	){
		return NULL;
	}

	for(int i = 0; file_list[i] != NULL; i++){
		fprintf(stderr, "%i:\n", i);
		for(int j = 0; file_list[i][j] != NULL; j++){
			fprintf(stderr, "\t%s\n", file_list[i][j]);
		}
	}

	// fprintf(stderr, "LIST:%s\n", PyBytes_AS_STRING(PyUnicode_AsEncodedString(PyObject_Repr(file_list),  "UTF-8", "strict")));

	// if(init(path, samplerate_hz, clip_len_ms) != 0){
	// 	PyErr_SetString(PyExc_RuntimeError, "arsd init failed");
	// 	PyErr_Occurred();
	// }

	Py_RETURN_NONE;
}

static PyObject* py_arsd_draw(PyObject *self, PyObject *args){
	float* output;
	int64_t output_size;
	timer(if(BLOCKING_draw_clip(&output, &output_size) != 0){
		PyErr_SetString(PyExc_RuntimeError, "Could not draw clip");
		PyErr_Occurred();
	}, BLOCKING_draw_clip);

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