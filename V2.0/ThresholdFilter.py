import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

threshold = 0.3
mu, std = .5, 0.2
dist = np.random.normal(mu, std, 1000000)
# dist = np.asarray(distribution)
low_vals = dist < threshold
dist[low_vals] = 0
plt.hist(dist, bins=2000)
plt.ylim(ymax=00)
plt.show()
