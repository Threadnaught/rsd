#include <inttypes.h>

#define max_file_len 100
#define max_batch_size 1000
#define max_sets 5
#define max_backlog 20
#define max_threads 10

struct arsd_config{
	int32_t samplerate_hz;
	int32_t clip_len_ms;
	int32_t clip_len_samples;
	int32_t run_in_samples;

	int32_t batch_size;
	int32_t set_count;
	int32_t backlog_depth;
	int32_t thread_count;
	int32_t pass_set_i;
};

#define arsd_config_t struct arsd_config

// api:
int32_t pick_batch(int32_t set_i, char** dest);

// scheduler:
int32_t init_scheduler(arsd_config_t* config);
int32_t BLOCKING_draw_batch(int32_t set_i, float* output);
int32_t NONBLOCKING_draw_batch(int32_t set_i, float** output);

// decoder:
int32_t init_decoder(arsd_config_t* config);
int32_t BLOCKING_draw_clip(char* filename, float* output_buffer);