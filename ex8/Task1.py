from comet_ml import Experiment
import numpy as np
import dlipr
import speckles
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
Try to remove the speckles from the test images.
Set up models for a flat and a deep autoencoder.
Train the deep autoencoder with and without shortcut connections!
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


(x_train_noisy, x_train), (x_test_noisy, x_test) = speckles.load_data()

x_train_noisy = preprocess_data(x_train_noisy, x_train)
x_train = preprocess_data(x_train, x_train)
x_test_noisy = preprocess_data(x_test_noisy, x_test)
x_test = preprocess_data(x_test, x_test)

speckles.plot_examples(x_test_noisy, x_test, fname='speckle_examples.png')
print(x_train_noisy.shape)

# some indices of interesting test data (it is not necessary to inspect all)
n = [1, 2, 4, 6, 10, 12, 14, 16, 22, 25, 27, 28, 30, 32, 35, 37, 39, 40, 41, 48, 49, 50, 52, 57, 61, 63, 64, 67, 70, 76]
