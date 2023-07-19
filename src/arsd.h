#include <inttypes.h>

#define max_file_len 100
#define max_batch_size 1000
#define max_sets 5
#define max_backlog 20
#define max_threads 48

struct arsd_config{
	int32_t samplerate_hz;
	int32_t clip_len_samples;
	int32_t run_in_samples;

	int32_t batch_size;
	int32_t set_count;
	int32_t backlog_depth;
	int32_t thread_count;
	int32_t pass_set_i;
	int32_t rng_seed;
	int32_t verbose;
};

#define arsd_config_t struct arsd_config

// api:
int32_t pick_batch(int32_t set_i, char** dest);

// scheduler:
int32_t init_scheduler(arsd_config_t* config_in, uint32_t* blocking_rng_state);
int32_t BLOCKING_draw_batch(int32_t set_i, float* output, uint32_t* rng_state);
int32_t NONBLOCKING_draw_batch(int32_t set_i, float** output);

// decoder:
int32_t init_decoder(arsd_config_t* config);
int32_t BLOCKING_draw_clip(char* filename, float* output_buffer, uint32_t* rng_state);

//Need this everywhere:
#define timer(instruction, short_name) {\
	struct timeval start, end; \
	int64_t diff; \
	gettimeofday(&start, NULL); \
	{instruction;} \
	gettimeofday(&end, NULL); \
	diff = ((end.tv_sec - start.tv_sec) * 1000 * 1000) + ((end.tv_usec - start.tv_usec)); \
	fprintf(stderr, "%s took %li us\n", #short_name, diff); \
}
