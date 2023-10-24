import matplotlib.pyplot as plt
import numpy as np
import rsd
import soundfile as sf
import os
import stat
import re
import datetime
import subprocess

def find(path, pattern):
	if stat.S_ISDIR(os.stat(path).st_mode):
		ret = []
		for el in os.listdir(path):
			ret = ret + find(os.path.join(path, el), pattern)
		return ret
	if re.match(pattern, path):
		return [path]

	return []

all_files = np.asarray(find('../../datasets/fma_full', '.*\\.mp3'))

validation_split = len(all_files) // 16
validation_shuffler = np.random.default_rng(0) #Make sure we always get the same split
validation_shuffler.shuffle(all_files)

validation_files = np.asarray(all_files[:validation_split])
train_files = np.asarray(all_files[validation_split:])

def pick_batch(batch_size, set_i):
	chosen_set = train_files if set_i == 0 else validation_files
	ret = chosen_set[np.random.choice(len(chosen_set), [batch_size])]
	return ret

rsd.init(pick_batch, 100, 2, thread_count=12, verbose=True, samplerate_hz=44100, clip_len_samples=44100)

# clip, seek_point_samples = rsd.BLOCKING_draw_clip(validation_files[0])
# sf.write('test-out.wav', clip, 44100)
# print(validation_files[0], seek_point_samples / 44100.0)

samples, names, seek_pts = rsd.draw_batch(0)

seek_pts_s = np.float32(seek_pts) / 44100.0

fig, axs = plt.subplots(8, 4)

for i in range(16):
	ax_i = i * 2
	x, y = ax_i // 4, ax_i % 4
	axs[x, y].plot(samples[i], np.arange(44100))

	test = subprocess.check_output(['/bin/bash', '-c', 'ffmpeg -ss %f -i %s -t 1 -f s16le -acodec pcm_s16le pipe:1' % (seek_pts_s[i], names[i].decode('utf-8'))])
	decode = np.frombuffer(test, dtype=np.int16)
	
	#redifine varaibles booo
	ax_i = (i * 2) + 1
	x, y = ax_i // 4, ax_i % 4
	
	axs[x, y].plot(decode[:44100], np.arange(44100))

plt.savefig('validation.png')