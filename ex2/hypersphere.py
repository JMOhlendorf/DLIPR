"""
---------------------------------------------------
Exercise 2 - Hypersphere
---------------------------------------------------
In this classification task the data consists of 4D vectors (x1, x2, x3, x4) uniformly sampled in each dimension between (-1, +1).
The data samples are classified according to their 2-norm as inside a hypersphere (|x|^2 < R) or outside (|x|^2 > R).
The task is to train a network to learn this classification based on a relatively small and noisy data set.
For monitoring the training and validating the trained model, we are going to split the dataset into 3 equal parts.
"""
from __future__ import print_function
from comet_ml import Experiment
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
models = keras.models
layers = keras.layers


# Set up YOUR experiment - login to comet, create new project (for new exercise)
# and copy the statet command
# or just change the name of the workspace, and the API (you can find it in the settings)
experiment = Experiment(api_key="EnterYourAPIKey",
                        project_name="exercise2", workspace="EnterGroupWorkspaceHere")

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
np.random.seed(1337)  # for reproducibility

n = 600  # number of data samples
nb_dim = 4  # number of dimensions
R = 1.1  # radius of hypersphere
xdata = 2 * np.random.rand(n, nb_dim) - 1  # features
ydata = np.sum(xdata**2, axis=1)**.5 < R  # labels, True if |x|^2 < R^2, else False

# add some normal distributed noise with sigma = 0.1
xdata += 0.1 * np.random.randn(n, nb_dim)

# turn class labels into one-hot encodings
# 0 --> (1, 0), outside of sphere
# 1 --> (0, 1), inside sphere
y1h = np.stack([~ydata, ydata], axis=-1)

# split data into training, validation and test sets of equal size
n_split = n // 3  # 1/3 of the data
X_train, X_valid, X_test = np.split(xdata, [n_split, 2 * n_split])
y_train, y_valid, y_test = np.split(y1h, [n_split, 2 * n_split])

print("  Training set, shape =", np.shape(X_train), np.shape(y_train))
print("Validation set, shape =", np.shape(X_valid), np.shape(y_valid))
print("      Test set, shape =", np.shape(X_test), np.shape(y_test))


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
#
# TODO: Specify a network with 4 hidden layers of 10 neurons each (ReLU)
# and an output layer (how many nodes?) with softmax activation.
#
model = models.Sequential()

model.compile(
    loss='categorical_crossentropy',
    optimizer='SGD',
    metrics=['accuracy'])

fit = model.fit(
    X_train, y_train,    # training data
    batch_size=n_split,  # no mini-batches, see next lecture
    epochs=4000,       # number of training epochs
    verbose=2,           # verbosity of shell output (0: none, 1: high, 2: low)
    validation_data=(X_valid, y_valid),  # validation data
    callbacks=[])        # optional list of functions to be called once per epoch

# print a summary of the network layout
print(model.summary())


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

#
# TODO: Obtain training, validation and test accuracy.
# You can use [loss, accuracy] = model.evaluate(X, y, verbose=0)
# Compare with the exact values using your knowledge of the Monte Carlo truth.
# (due to noise the exact accuracy will be smaller than 1)
# Locate the best stopping point.
#


#
# TODO: Plot training history in terms of loss and accuracy
# You can obtain these values from the fit.history dictionary.
#
print(fit.history.keys())
print(fit.history['acc'])


#
# TODO: Train for a number of epochs corresponding to the best stopping point.
# You can use the EarlyStopping callback for this:
# earlystopping = keras.callbacks.EarlyStopping(patience=1)
#
