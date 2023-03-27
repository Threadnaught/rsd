#include<string.h>
#include<stdio.h>

#include "arsd.h"

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

static char* path;
static int samplerate_hz;
static int clip_len_samples;
static int inited = 0;

int init(char* path_in, int samplerate_hz_in, int clip_len_ms_in){
	int clip_length_samplerate_product;
	//Trap attepmpts to re-init
	if(inited){
		return -1;
	}
	inited = 1;
	//Set trivial variables
	path = path_in;
	samplerate_hz = samplerate_hz_in;
	
	//For this case, I think we should warn and keep moving
	clip_length_samplerate_product = samplerate_hz_in * clip_len_ms_in;
	if((clip_length_samplerate_product % 1000) != 0){
		fprintf(stderr, "WARNING: requested clip length does not evenly divide into samples. Continuing.");
	}
	clip_len_samples = clip_length_samplerate_product / 1000;

	return 0;
}

int BLOCKING_draw_clip(float** output, int* output_samples){
	AVFormatContext* format_context = NULL;

	int chosen_stream = 0;
	
	const AVCodec* decoder;
	AVCodecContext* decoder_context;

	AVPacket* packet;
	AVFrame* frame;

	float* sample_buffer;
	int cursor = 0;

	return -1;
}