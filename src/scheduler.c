#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "arsd.h"

// PyObject* py_arsd_draw(PyObject *self, PyObject *args){
// 	float* output = (float*)malloc(batch_size * clip_len_samples * sizeof(float));

// 	char batch_filenames[max_batch_size][max_file_len];
// 	char* batch_filename_ptrs[max_batch_size]; //TODO: got to be a better way to do this
// 	for(int i = 0; i < max_batch_size; i++) batch_filename_ptrs[i] = batch_filenames[i];
// 	pick_batch(0, batch_filename_ptrs);
	
// 	timer(for(int i = 0; i < batch_size; i++){
// 		fprintf(stderr, "file:%s\n", batch_filenames[i]);

// 		if(BLOCKING_draw_clip(batch_filenames[i], output + (clip_len_samples * i)) != 0){
// 			PyErr_SetString(PyExc_RuntimeError, "Could not draw clip");
// 			PyErr_Occurred();
// 		}

// 	}, draw_clips);
	
// 	npy_intp dims[2] = {batch_size, clip_len_samples};

// 	PyObject* arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, output);
// 	PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA); // TODO: there has been some debate over wheter this is a correct dellocator

// 	return arr;
// }




int init_scheduler(int set_count, int batch_size, int backlog){
	return 0;
}
