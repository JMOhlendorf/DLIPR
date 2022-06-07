from comet_ml import Experiment
import numpy as np
import densenet
import dlipr
from tensorflow import keras
image = keras.preprocessing.image


# Set up YOUR experiment - login to comet, create new project (for new exercise)
# and copy the statet command
# or just change the name of the workspace, and the API (you can find it in the settings)
experiment = Experiment(api_key="EnterYourAPIKey",
                        project_name="exercise6", workspace="EnterGroupWorkspaceHere")

# -----------------------------------------
# Data
# -----------------------------------------
data = dlipr.cifar.load_cifar10()

# data sets, hold out 4000 training images for validation
X_train, X_valid = np.split(data.train_images, [-4000])
y_train, y_valid = np.split(data.train_labels, [-4000])
X_test = data.test_images
y_test = data.test_labels

# simple input scaling (for use with provided weights)
X_train = X_train / 255.
X_valid = X_valid / 255.
X_test = X_test / 255.

Y_train = dlipr.utils.to_onehot(y_train)
Y_valid = dlipr.utils.to_onehot(y_valid)
Y_test = dlipr.utils.to_onehot(y_test)


# -----------------------------------------
# Model and training
# -----------------------------------------
model = densenet.DenseNet(...)
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=["accuracy"])

# data augmentation
generator = image.ImageDataGenerator(
    width_shift_range=4. / 32,
    height_shift_range=4. / 32,
    fill_mode='constant',
    horizontal_flip=True)

# fit using augmented data
model.fit_generator(
    generator.flow(X_train, Y_train, batch_size=64),
    steps_per_epoch=len(X_train) // 64,
    validation_data=(X_valid, Y_valid),
    epochs=300,
    verbose=2)


# -----------------------------------------
# Evaluation
# -----------------------------------------
