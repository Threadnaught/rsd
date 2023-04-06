#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "arsd.h"

#define locking(mut, code) {\
	pthread_mutex_lock(&mut);\
	code;\
	pthread_mutex_unlock(&mut);\
}

//The ARSD decode control flow looks like this:
//	[Python Thread]
//	NONBLOCKING_draw_batch:
//		Loop:
//			Lock Mutex:
//				Fill any relevant batches in the `needs_filenames` state
//				If there is a relevant batch in `decoded` state:
//					Set the batch to `needs_filenames`
//					return batch
//	
//	[Worker Thread]
//	Loop:
//		Lock Mutex:
//			Find a batch in the `ready_to_decode` state
//		If there is a batch available:
//			Decode batch files, one by one
//			Lock Mutex:
//				If decode is succesful:
//					Set batch to to `decoded`
//				Else:
//					Cleanup, and set batch to `needs_filenames`

enum batch_status{
	needs_filenames,
	ready_to_decode,
	decoding,
	decoded
};
#define batch_status_t enum batch_status

static arsd_config_t* config;

static float* completed_batches[max_sets][max_backlog];
static char batch_file_names[max_sets][max_backlog][max_batch_size][max_file_len];
static batch_status_t batch_statuses[max_sets][max_backlog];

pthread_mutex_t common_lock;

// I love / hate C
void sleep_ms(int32_t ms){
	struct timespec sleep_interval, rem;
	sleep_interval.tv_sec = 0;
	sleep_interval.tv_nsec = ms * 1000;
	if(nanosleep(&sleep_interval, &rem) != 0)
		exit(1);
}

int BLOCKING_draw_batch(int set_i, float* output){
	char batch_filenames[max_batch_size][max_file_len];

	char* batch_filename_ptrs[max_batch_size];
	for(int i = 0; i < max_batch_size; i++) batch_filename_ptrs[i] = batch_filenames[i];
	
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

void* worker_thread(void* unused){
	while(1){
		int depth = -1;
		int set = -1;
		int should_decode = 0;
		int decode_failed = 0;
		locking(common_lock, {
			// Prefer one for each set, not all in one set and none in the others
			for(depth = 0; depth < config->backlog_depth && (!should_decode); depth++){
				for(set = 0; set < config->set_count && (!should_decode); set++){
					if(batch_statuses[set][depth] == ready_to_decode){
						// fprintf(stderr, "Set decoding %i %i\n", set, depth);
						batch_statuses[set][depth] = decoding;
						completed_batches[set][depth] = (float*)malloc(config->batch_size * config->clip_len_samples * sizeof(float));
						should_decode = 1;
					}
				}
			}
		});
		if(should_decode){
			set -= 1;
			depth -= 1;
			for(int i = 0; i < config->batch_size; i++){
				// fprintf(stderr, "Decoding %i %i (%s ...)\n", set, depth, (char*)batch_file_names[set][depth][i]);
				if(BLOCKING_draw_clip(
					batch_file_names[set][depth][i],
					completed_batches[set][depth] + (config->clip_len_samples * i)
				) != 0){
					fprintf(stderr, "Discarding entire batch due to %s decode failure\n", batch_file_names[set][depth][i]);
					// TODO: link to GH here
					fprintf(stderr, "See arsd github for details of how to normalize your input files.\n");
					decode_failed = 1;
					break;
				}
			}

			locking(common_lock, {
				if(decode_failed){
					// fprintf(stderr, "Decode failed\n");
					free(completed_batches[set][depth]);
					batch_statuses[set][depth] = needs_filenames;
				} else {

					// fprintf(stderr, "Decoded\n");
					batch_statuses[set][depth] = decoded;
				}
			});
		} else {
			sleep_ms(1);
		}
	}
	return NULL;
}

int NONBLOCKING_draw_batch(int set_i, float** output){
	*output = NULL;

	while(1){
		// Picking batch is inside this loop, as a file may fail to decode, requring new filenames to be picked
		// While not exactly likely, this can happen multiple times.
		locking(common_lock, {
			for(int depth = 0; depth < config->backlog_depth; depth++){
				if(batch_statuses[set_i][depth] == needs_filenames){
					char* batch_filename_ptrs[max_batch_size];
					for(int i = 0; i < max_batch_size; i++)
						batch_filename_ptrs[i] = batch_file_names[set_i][depth][i];
					while(pick_batch(set_i, batch_filename_ptrs) != 0);
					// fprintf(stderr, "Ready To Decode %i %i (%s ...)\n", set_i, depth, batch_file_names[set_i][depth][0]);
					batch_statuses[set_i][depth] = ready_to_decode;
				}
			}
			
			for(int depth = 0; depth < config->backlog_depth; depth++){
				if(batch_statuses[set_i][depth] == decoded){
					*output = completed_batches[set_i][depth];
					// fprintf(stderr, "Needs filenames (returning)\n");
					batch_statuses[set_i][depth] = needs_filenames;
					break;
				}
			}
		});

		if((*output) != NULL)
			return 0;

		sleep_ms(1);
	}
}

int init_scheduler(arsd_config_t* config_in){
	config = config_in;
	for(int set = 0; set < config->set_count; set++){
		for(int depth = 0; depth < config->backlog_depth; depth++){
			batch_statuses[set][depth] = needs_filenames;
		}
	}

	pthread_t threads[max_threads];
	for(int i = 0; i < config->thread_count; i++) pthread_create(threads+i, NULL, worker_thread, NULL);
	return 0;
}
