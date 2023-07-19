# ARSD

Audio Repetitive Sampling Decoder (ARSD) is a wickedly fast MP3 decoding library for Python/ML, made with libavcodec (FFMpeg). In the time that it took you to read that sentence, ARSD could have easily decoded an hour of audio.

ARSD's multithreaded architecture allows the next batches to be CPU decoded in the background while the GPU is busy doing other things. FFMpeg's seek implementation allows for efficient random access of clips from longer audio files.

## How do I set it up?

### Install dependencies

If you are on debian/ubuntu, you should be able to run `apt install ffmpeg libavutil-dev libavcodec-dev libavformat-dev libavfilter-dev`.

If you are on other linux or osx, these packages should be available but may have different names.

If you are on windows, may god have mercy on your soul.

### Install arsd

Now you should be able to run `make install`

## How do I use it?

Like this:
```python
import numpy as np
import arsd

# This function is your batch picker - responsible for providing filenames to decode
files = np.asarray(['file1.mp3', 'file2.mp3'])
def pick_batch(batch_size):
	return files[np.random.choice(len(files), [batch_size])]

#Second argument is your batch size
arsd.init(pick_batch, 10)

while True:
	batch = arsd.draw_batch()
	print(batch.shape) #Do something with the batch
```

### What about training validation splits?

Right here:
```python
import numpy as np
import arsd

files = np.asarray(['file1.mp3', 'file2.mp3'])
files_validation = np.asarray(['file3.mp3', 'file4.mp3'])

def pick_batch(batch_size, set_i): #notice the extra set_i argument - either 0 or 1
	if set_i == 0:
		return files[np.random.choice(len(files), [batch_size])]
	else:
		return files_validation[np.random.choice(len(files_validation), [batch_size])]

arsd.init(pick_batch, 10, 2) #extra argument - how many sets do you need?

while True:
	#The argument argument is passed into set_i of batch picker
	batch = arsd.draw_batch(0)
	validation_batch = arsd.draw_batch(1)
```

## File Normalization

_Note:_ ARSD is still under active development, so you'll probably see this periodically even for correctly normalized files.

ARSD expects audio files to be uniform in format, sample rate and channel count for maximum decode rate. Use [normalize-inplace.sh](./scripts/normalize-inplace.sh).
