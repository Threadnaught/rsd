#include <inttypes.h>

int init(char* path_in, int samplerate_hz_in, int clip_len_ms_in);
int BLOCKING_draw_clip(float** output, int64_t* output_samples);