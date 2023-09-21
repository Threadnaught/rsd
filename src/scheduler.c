#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "arsd.h"
#include <math.h>

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

pthread_t threads[max_threads];
uint32_t rng_states[max_threads];

pthread_mutex_t common_lock;

// I love / hate C
void sleep_ms(int32_t ms){
	struct timespec sleep_interval, rem;
	sleep_interval.tv_sec = 0;
	sleep_interval.tv_nsec = ms * 1000;
	if(nanosleep(&sleep_interval, &rem) != 0)
		exit(1);
}

int32_t BLOCKING_draw_batch(int32_t set_i, float* output, uint32_t* rng_state){
	char batch_filenames[max_batch_size][max_file_len];

	char* batch_filename_ptrs[max_batch_size];
	for(int32_t i = 0; i < max_batch_size; i++) batch_filename_ptrs[i] = batch_filenames[i];
	
	if(pick_batch(set_i, batch_filename_ptrs) != 0)
		return -1;
	

	for(int32_t i = 0; i < config->batch_size; i++){
		// fprintf(stderr, "file:%s\n", batch_filenames[i]);

		if(BLOCKING_draw_clip(batch_filenames[i], output + (config->clip_len_samples * i), rng_state) != 0){
			return -1;
		}
	}
	return 0;
}

void* worker_thread(void* rng_state_uncast){
	uint32_t* rng_state = (uint32_t*)rng_state_uncast;

	// fprintf(stderr, "WORKER THREAD STARTED WITH SEED %i\n", *rng_state);

	while(1){
		int32_t depth = -1;
		int32_t set = -1;
		int32_t should_decode = 0;
		int32_t decode_failed = 0;

		locking(common_lock, {
			// Prefer one for each set, not all in one set and none in the others
			for(depth = 0; depth < config->backlog_depth && (!should_decode); depth++){
				for(set = 0; set < config->set_count && (!should_decode); set++){
					if(batch_statuses[set][depth] == ready_to_decode){
						// fprintf(stderr, "Set decoding %i %i\n", set, depth);
						batch_statuses[set][depth] = decoding;
						completed_batches[set][depth] = (float*)malloc(config->batch_size * config->clip_len_samples * sizeof(float));

						// TEMPORARY - force NaN to be in all batches with gaps TODO remove
						for(int i = 0; i < config->batch_size * config->clip_len_samples; i++)
							completed_batches[set][depth][i] = -sqrt(-1);

						should_decode = 1;
					}
				}
			}
		});
		if(should_decode){
			set -= 1;
			depth -= 1;
			for(int32_t i = 0; i < config->batch_size; i++){
				// fprintf(stderr, "Decoding %i %i (%s ...)\n", set, depth, (char*)batch_file_names[set][depth][i]);
				if(BLOCKING_draw_clip(
					batch_file_names[set][depth][i],
					completed_batches[set][depth] + (config->clip_len_samples * i),
					rng_state
				) != 0){
					fprintf(stderr, "Discarding entire batch due to %s decode failure\n", batch_file_names[set][depth][i]);
					fprintf(stderr, "See https://github.com/Threadnaught/arsd#file-normalization for details of how to normalize your input files.\n");
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

					for(int i = 0; i < config->batch_size * config->clip_len_samples; i++)
						if(isnan(completed_batches[set][depth][i])) {
							fprintf(stderr, "BAD DECODE starts at %i/%i\n", i, config->batch_size * config->clip_len_samples);
							break;
						}

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

int32_t NONBLOCKING_draw_batch(int32_t set_i, float** output_samples, char** output_filenames){
	*output_samples = NULL;

	while(1){
		// Picking batch is inside this loop, as a file may fail to decode, requring new filenames to be picked
		// While not exactly likely, this can happen multiple times.
		locking(common_lock, {
			for(int32_t depth = 0; depth < config->backlog_depth; depth++){
				if(batch_statuses[set_i][depth] == needs_filenames){
					char* batch_filename_ptrs[max_batch_size];
					for(int32_t i = 0; i < max_batch_size; i++)
						batch_filename_ptrs[i] = batch_file_names[set_i][depth][i];
					if(pick_batch(set_i, batch_filename_ptrs) != 0){
						//unlock mutex early
						pthread_mutex_unlock(&common_lock);
						return -1;
					}
					// fprintf(stderr, "Ready To Decode %i %i (%s ...)\n", set_i, depth, batch_file_names[set_i][depth][0]);
					batch_statuses[set_i][depth] = ready_to_decode;
				}
			}
			
			for(int32_t depth = 0; depth < config->backlog_depth; depth++){
				if(batch_statuses[set_i][depth] == decoded){
					*output_samples = completed_batches[set_i][depth];
					for(int i = 0; i < max_batch_size; i++){
						memcpy(output_filenames[i], batch_file_names[set_i][depth][i], max_file_len);
					}
					// fprintf(stderr, "Needs filenames (returning)\n");
					batch_statuses[set_i][depth] = needs_filenames;
					break;
				}
			}
		});

		if((*output_samples) != NULL){
			return 0;
		}

		sleep_ms(1);
	}
}

int32_t init_scheduler(arsd_config_t* config_in, uint32_t* blocking_rng_state){
	config = config_in;
	for(int32_t set = 0; set < config->set_count; set++){
		for(int32_t depth = 0; depth < config->backlog_depth; depth++){
			batch_statuses[set][depth] = needs_filenames;
		}
	}


	for(int32_t i = 0; i < config->thread_count; i++) {
		rng_states[i] = rand_r(blocking_rng_state);
		pthread_create(threads+i, NULL, worker_thread, rng_states+i);
	}
	return 0;
}
