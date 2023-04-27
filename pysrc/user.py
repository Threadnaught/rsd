import matplotlib.pyplot as plt
import numpy as np
import arsd
import soundfile as sf
import os
import stat
import re
import datetime

def find(path, pattern):
	if stat.S_ISDIR(os.stat(path).st_mode):
		ret = []
		for el in os.listdir(path):
			ret = ret + find(os.path.join(path, el), pattern)
		return ret
	if re.match(pattern, path):
		return [path]

	return []

all_files = np.asarray(find('/mnt/datasets/fma_full', '.*\\.mp3'))

validation_split = len(all_files) // 16
validation_shuffler = np.random.default_rng(0) #Make sure we always get the same split
validation_shuffler.shuffle(all_files)

validation_files = np.asarray(all_files[:validation_split])
train_files = np.asarray(all_files[validation_split:])

def pick_batch(batch_size, set_i):
	chosen_set = train_files if set_i == 0 else validation_files
	ret = chosen_set[np.random.choice(len(chosen_set), [batch_size])]
	return ret

arsd.init(pick_batch, 100, 2, thread_count=5)

while True:
	start = datetime.datetime.utcnow()
	for _ in range(100):
		data = arsd.draw_batch(0)
	data = arsd.draw_batch(1)
	end = datetime.datetime.utcnow()
	
	print(end, ' shape:', data.shape, 'time per:', end - start)

