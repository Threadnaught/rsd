#include<stdio.h>
#include<assert.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

int main(){
	AVFormatContext* format_context = NULL;

	int chosen_stream;
	
	const AVCodec* decoder;
	AVCodecContext* decoder_context;

	long int duration;
	AVRational timebase;

	AVPacket* packet;
	AVFrame* frame;

	fprintf(stderr, "starting\n");

	// Open input, find stream info, select best one
	assert(avformat_open_input(&format_context, "samples/000002.mp3", NULL, NULL) == 0);
	assert(avformat_find_stream_info(format_context, NULL) == 0);
	assert((chosen_stream = av_find_best_stream(format_context, AVMEDIA_TYPE_AUDIO, -1, -1, &decoder, 0)) >= 0);

	av_dump_format(format_context, 0, "", 0);

	// Prepare to decode
	assert((decoder_context = avcodec_alloc_context3(decoder)) != NULL);
	assert(avcodec_open2(decoder_context, decoder, NULL) == 0);

	// Alloc frame and packet:
	assert((packet = av_packet_alloc()) != NULL);
	assert((frame = av_frame_alloc()) != NULL);

	duration = format_context->streams[chosen_stream]->duration;
	timebase = format_context->streams[chosen_stream]->time_base;

	int64_t seek_target = 211680000;

	assert(avformat_seek_file(
		format_context,
		chosen_stream,
		seek_target - 1000000,
		seek_target,
		seek_target + 1000000,
		0
	) == 0);
	
	fprintf(stderr, 
		"duration:%li, timebase %i/%i\n",
		duration,
		timebase.num,
		timebase.den
	);
	// return 0;

	// The fun bit:
	while (av_read_frame(format_context, packet) == 0) {
		if (packet->stream_index != chosen_stream){
			// This is not the correct frame, skip it
			av_packet_unref(packet);
			continue;
		}

		assert(avcodec_send_packet(decoder_context, packet) == 0);

		while(avcodec_receive_frame(decoder_context, frame) == 0){
			fwrite(
				frame->data[0],
				frame->nb_samples,
				av_get_bytes_per_sample(decoder_context->sample_fmt),
				stdout
			);
		}
		av_packet_unref(packet);
	}

	av_packet_free(&packet);
	av_frame_free(&frame);

	avformat_close_input(&format_context);
	avcodec_free_context(&decoder_context);

	

	fprintf(stderr, "finished\n");

}