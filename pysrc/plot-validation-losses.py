import numpy as np
import matplotlib.pyplot as plt
import numpy as np

all_losses = np.load('all_losses.npy')

all_losses = np.reshape(all_losses, [-1, 128])

plt.hist([np.reshape(all_losses, -1), np.mean(all_losses, -1)], bins=100)
plt.yscale('log')
plt.show()