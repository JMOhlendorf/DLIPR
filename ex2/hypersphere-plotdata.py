import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1337)

n = 600  # number of data samples
nb_dim = 4  # number of dimensions
xdata = 2 * np.random.rand(n, nb_dim) - 1  # features
ydata = np.sum(xdata**2, axis=1) < 1.1**2  # labels, True if |x|^2 < 1, else False

# add some normal distributed noise with sigma = 0.1
xdata += 0.1 * np.random.randn(n, nb_dim)


# Plot data (for 2 dimensions)
fig, ax = plt.subplots(1)

r = np.sum(xdata**2, axis=1)**.5
ax.hist([r[~ydata], r[ydata]],
        label=('outside', 'inside'),
        bins=np.linspace(0, 2, 21), histtype='stepfilled', alpha=0.5)

ax.legend()
ax.axvline(1.1, color='r')
ax.set(xlabel='$\Vert x \Vert_2$', ylabel='N')
ax.grid(True)
fig.savefig('hypersphere-data.png')
plt.show()
