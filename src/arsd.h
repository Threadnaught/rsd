#include <inttypes.h>

#define max_file_len 100
#define max_batch_size 500

// api:
// int pick_batch(int set_i, char* dest);

// decoder:
int init(int samplerate_hz_in, int clip_len_ms_in, int run_in_samples);
int BLOCKING_draw_clip(char* filename, float** output, int64_t* output_samples);