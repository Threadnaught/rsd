# RSD

Repetitive Sampling Decoder (RSD) is a wickedly fast MP3 decoding library for Python/ML, made with libavcodec (FFMpeg). In the time that it took you to read that sentence, RSD could have easily decoded an hour of audio.

RSD's multithreaded architecture allows the next batches to be CPU decoded in the background while the GPU is busy doing other things. FFMpeg's seek implementation allows for efficient random access of clips from longer audio files.

## How do I set it up?

### Install dependencies

#### System packages
If you are on debian/ubuntu, you should be able to run

```
apt install ffmpeg libavutil-dev libavcodec-dev libavformat-dev libavfilter-dev
```

On arch, this should work:

```
pacman -S ffmpeg
```

If you are on other linux or osx, these packages should be available but may have different names.

If you are on windows, may god have mercy on your soul.

### Pip packages

```
pip install numpy
```

### Install rsd

Now you should be able to run `make install`

## How do I use it?

Take a look at `pysrc/user.py` for a minimum viable usage.

## File Normalization

_Note:_ RSD is still under active development, so you'll probably see this error periodically even for correctly normalized files.

RSD expects audio files to be uniform in format, sample rate and channel count for maximum decode rate. Use [normalize-inplace.sh](./scripts/normalize-inplace.sh).
