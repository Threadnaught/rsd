#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "arsd.h"

static arsd_config_t* config;

int BLOCKING_draw_batch(int set_i, float* output){
	char batch_filenames[max_batch_size][max_file_len];

	char* batch_filename_ptrs[max_batch_size]; //TODO: got to be a better way to do this
	for(int i = 0; i < max_batch_size; i++) batch_filename_ptrs[i] = batch_filenames[i];
	
	// TODO: raise this exception to calling code / print it out
	if(pick_batch(set_i, batch_filename_ptrs) != 0)
		return -1;
	

	for(int i = 0; i < config->batch_size; i++){
		// fprintf(stderr, "file:%s\n", batch_filenames[i]);

		if(BLOCKING_draw_clip(batch_filenames[i], output + (config->clip_len_samples * i)) != 0){
			return -1;
		}
	}
	return 0;
}

int init_scheduler(arsd_config_t* config_in){
	config = config_in;
	return 0;
}
