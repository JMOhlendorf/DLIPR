import numpy as np
import matplotlib.pyplot as plt
import dlipr


# load and prepare dataset
data = dlipr.ising.load_data()

T = data.classes
Tc = 2.27
y_test = T[data.test_labels] > Tc
Y_test = dlipr.utils.to_onehot(y_test)
X_test = data.test_images[..., np.newaxis]


# plot average magnetization vs temperature
M = np.abs(np.mean(X_test, axis=(1, 2, 3)))  # magnetization for each sample
M_mean = np.zeros_like(T)
M_std = np.zeros_like(T)
print(data.test_labels)
print(len(data.test_labels))
for i in range(len(T)):
    print(i)
    idx = data.test_labels == i
    M_mean[i] = np.mean(M[idx])
    M_std[i] = np.std(M[idx])

plt.figure()
plt.errorbar(T, M_mean, yerr=M_std, fmt='o')
plt.axvline(x=Tc, color='k', linestyle='--', label='Tc')
plt.xlabel('T')
plt.ylabel('|M(T)|')
plt.grid()
plt.savefig('./magnetization.png', bbox_inches='tight')
