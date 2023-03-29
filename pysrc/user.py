import matplotlib.pyplot as plt
import numpy as np
import arsd
import soundfile as sf

arsd.init('samples/000002.mp3')

fig, (a,b) = plt.subplots(1,2)

data = arsd.BLOCKING_draw_clip()
# a.plot(np.arange(1000), data[:1000])
# b.plot(np.arange(1000), data[-1000:])
# plt.show()

sf.write('samples/clip.flac', data, 44100)

