from comet_ml import Experiment
import numpy as np
import dlipr
import speckles
import matplotlib.pyplot as plt
from tensorflow import keras
models = keras.models
layers = keras.layers
backend = keras.backend


# Set up YOUR experiment - login to comet, create new project (for new exercise)
# and copy the statet command
# or just change the name of the workspace, and the API (you can find it in the settings)
#experiment = Experiment(api_key="EnterYourAPIKey",
#                        project_name="exercise8", workspace="EnterGroupWorkspaceHere")


"""
Calculate the reconstructions of measured scattering patterns with your trained
models.
"""


def preprocess_data(data, norm_data):
    # logarithmic intensity values
    data = np.log10(data + 0.01)
    norm_data = np.log10(norm_data + 0.01)
    # norm input data to max. value of undistorted scattering pattern
    max_val = np.max(norm_data, axis=1)
    data = data / max_val.reshape(len(max_val), 1)
    # reshape data for convolutional network
    data = np.reshape(data, (len(data), 64, 64, 1))
    # limit maximum intensity values
    data = np.clip(data, 0., 1.1)
    return data


# read in measured scattering patterns
real_names = ['cb019_100.npy', 'cb019_103.npy', 'AuCd_302_0K_H_III_1.npy']
real_imgs = np.empty((len(real_names), 64 * 64))

for i in range(len(real_names)):
    img = np.load(real_names[i])
    img = img[np.newaxis]
    real_imgs[i, :] = img

real_imgs = preprocess_data(real_imgs, real_imgs)
print('real_imgs.shape',real_imgs.shape)

plt.figure(figsize=(15,4))
for i in range(3):
    print('real_imgs[i].shape',real_imgs[i].shape)
    plt.imshow(real_imgs[i].reshape(64, 64))
    plt.savefig('pic{0}.png'.format(i))
    
    


