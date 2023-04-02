#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#include "arsd.h"

// static volatile char current_files[max_batch_size][max_file_len];
// static volatile float* current_decodes[max_batch_size];
// static volatile int current_set;

// int schedule(){
// 	struct timespec sleep_interval, rem;
// 	sleep_interval.tv_sec = 0;
// 	sleep_interval.tv_nsec = 100000;

// 	while(1){
// 		fprintf(stderr, "SCHEDULE\n");
		
// 		if(nanosleep(&sleep_interval, &rem) != 0){
// 			exit(0);
// 		}
// 	}
// 	return 0;
// }
// int worker(void* arg){
// 	struct timespec sleep_interval, rem;
// 	sleep_interval.tv_sec = 0;
// 	sleep_interval.tv_nsec = 100000;

// 	while(1){
// 		fprintf(stderr, "WORKER\n");
		
// 		if(nanosleep(&sleep_interval, &rem) != 0){
// 			exit(0);
// 		}
// 	}
// }



int init_scheduler(int set_count, int batch_size, int backlog){
	return 0;
}
