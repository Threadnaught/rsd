#include <inttypes.h>

#define max_file_len 100
#define max_batch_size 500
#define max_sets 5
#define max_backlog 10

// TODO: make all integer types (u)intxx_t

// api:
// int pick_batch(int set_i, char** dest);

// scheduler:
int init_scheduler(int set_count, int batch_size, int backlog);

// decoder:
int init_decoder(int samplerate_hz_in, int clip_len_ms_in, int run_in_samples, int64_t* clip_len_samples_out);
int BLOCKING_draw_clip(char* filename, float* output_buffer);