import matplotlib.pyplot as plt
import numpy as np
import arsd
import soundfile as sf
import os

def list_fullpath(base):
	return [os.path.join(base, x) for x in os.listdir(base)]

sets = [list_fullpath('samples/many'), list_fullpath('samples/many-2')]

def pick_file(set_i):
	pop = sets[set_i]
	return pop[np.random.randint(0, len(pop))]

arsd.init(pick_file)
# arsd.init('fuck')

# fig, (a,b) = plt.subplots(1,2)

# data = arsd.BLOCKING_draw_clip()
# # a.plot(np.arange(1000), data[:1000])
# # b.plot(np.arange(1000), data[-1000:])
# # plt.show()

# sf.write('samples/clip.flac', data, 44100)

