#include <inttypes.h>

#define max_file_len 100
#define max_batch_size 500
#define max_sets 5
#define max_backlog 10

// TODO: make all integer types (u)intxx_t

struct arsd_config{
	int32_t samplerate_hz;
	int32_t clip_len_ms;
	int32_t clip_len_samples;
	int32_t run_in_samples;

	int32_t batch_size;
	int32_t set_count;
	int32_t backlog;
};
#define arsd_config_t struct arsd_config

// api:
int pick_batch(int set_i, char** dest);

// scheduler:
int init_scheduler(arsd_config_t* config);
int BLOCKING_draw_batch(int set_i, float* output);

// decoder:
int init_decoder(arsd_config_t* config);
int BLOCKING_draw_clip(char* filename, float* output_buffer);