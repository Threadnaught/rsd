# NOTE: this file uses pytorch. it is not a dependency of rsd as a whole,
# only for validation

import rsd
import torch
from streamp3 import MP3Decoder
import stat
import os
import re
import numpy as np
import matplotlib.pyplot as plt

device='cuda' #set to 'CPU' to live life in the slow lane

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

def get_pcm_mp3(f):
	with open(f, 'rb') as file:
		decoder = MP3Decoder(file)
		output = []
		for chunk in decoder:
			chunk_enc = np.float32(np.frombuffer(chunk, dtype=np.int16))
			
			output.append(chunk_enc)
		return np.float32(np.reshape(output, -1)) / 32768

stft_frame_length = 4094
stft_shift = stft_frame_length // 8
log_epsillon = 1e-6 #to avoid a 0 becoming a NaN

def process_pcm(pcm):
	torchified = torch.from_numpy(pcm).to(device)

	stft = torch.stft(torchified, stft_frame_length, stft_shift, return_complex=True)
	log_mag =  torch.log(torch.abs(stft) + log_epsillon)
	return torchified, log_mag

bs = 128
epochs = 16
rsd.init(pick_batch, bs, 2, thread_count=12, verbose=True, samplerate_hz=44100, clip_len_samples=44100)

# rsd_pcm, names, seek_pts = rsd.draw_batch(0)
# for i in range(bs):
# 	ffmpeg_pcm = get_pcm_mp3(names[i].decode('utf-8'))

# 	ffmpeg_pcm = ffmpeg_pcm[seek_pts[i]:seek_pts[i]+44100]

# 	fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)

# 	ax1.plot(np.arange(44100), rsd_pcm[i])
# 	ax1.plot(np.arange(44100), ffmpeg_pcm, '--')
# 	ax2.plot(np.arange(44100), ffmpeg_pcm)

# 	rsd_pcm_torch, rsd_spec = process_pcm(rsd_pcm)
# 	ffmpeg_pcm_torch, ffmpeg_spec = process_pcm(ffmpeg_pcm)

# 	# ax2.imshow(rsd_spec[i].to('cpu').numpy(), cmap='nipy_spectral', aspect='auto')
# 	ax3.imshow(ffmpeg_spec.to('cpu').numpy(), cmap='nipy_spectral', aspect='auto')

# 	print((rsd_spec[i] - ffmpeg_spec).abs().mean())

# 	ax4.imshow((rsd_spec[i] - ffmpeg_spec).to('cpu').numpy(), cmap='nipy_spectral', aspect='auto')

# 	plt.show()

# 	break



losses = []

for j in range(epochs):
	try:
		rsd_pcm, names, seek_pts = rsd.draw_batch(0)
		ffmpeg_pcm = np.zeros(rsd_pcm.shape)
		for i in range(bs):
			ffmpeg_pcm_raw = get_pcm_mp3(names[i].decode('utf-8'))

			ffmpeg_pcm[i] = ffmpeg_pcm_raw[seek_pts[i]:seek_pts[i]+44100]
		
		rsd_pcm_torch, rsd_spec = process_pcm(rsd_pcm)
		ffmpeg_pcm_torch, ffmpeg_spec = process_pcm(ffmpeg_pcm)

		losses.append((rsd_spec - ffmpeg_spec).abs().mean([1,2]))
	except:
		print('batch failed')
	print('%i/%i' % (j+1, epochs))

all_losses = torch.stack(losses).flatten().to('cpu').numpy()

np.save('all_losses.npy', all_losses)