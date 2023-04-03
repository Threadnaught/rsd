#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "arsd.h"

static int clip_len_samples;
static int set_count;
static int batch_size;
static int backlog;

int BLOCKING_draw_batch(float* output){
	char batch_filenames[max_batch_size][max_file_len];

	char* batch_filename_ptrs[max_batch_size]; //TODO: got to be a better way to do this
	for(int i = 0; i < max_batch_size; i++) batch_filename_ptrs[i] = batch_filenames[i];
	
	pick_batch(0, batch_filename_ptrs);

	for(int i = 0; i < batch_size; i++){
		// fprintf(stderr, "file:%s\n", batch_filenames[i]);

		if(BLOCKING_draw_clip(batch_filenames[i], output + (clip_len_samples * i)) != 0){
			return -1;
		}
	}
	return 0;
}

int init_scheduler(int clip_len_samples_in, int set_count_in, int batch_size_in, int backlog_in){
	clip_len_samples = clip_len_samples_in;
	set_count = set_count_in;
	batch_size = batch_size_in;
	backlog = backlog_in;

	return 0;
}
