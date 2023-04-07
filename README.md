# ARSD

Audio Repetitive Sampling Decoder (ARSD) is a wickedly fast MP3 decoding library for Python/ML, made with FFMpeg. In the time that it took you to read that sentence, ARSD could have easily decoded an hour of audio.

ARSD's multithreaded architecture allows the next batches to be CPU decoded in the background while the GPU is busy doing other things. FFMpeg's seek implementation allows for efficient random access of clips from longer audio files.

## File Normalization

ARSD expects audio files to be uniform in format, sample rate and channel count for maximum decode rate. Use [normalize-inplace.sh](./scripts/normalize-inplace.sh).