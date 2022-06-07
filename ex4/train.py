from comet_ml import Experiment
import numpy as np
import matplotlib.pyplot as plt
import dlipr
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import os

models = keras.models
layers = keras.layers

# Set up YOUR experiment - login to comet, create new project (for new exercise)
# and copy the statet command
# or just change the name of the workspace, and the API (you can find it in the settings)
experiment = Experiment(api_key="jQH3I2Zcq46PU16iZDsMtfYIS",
                        project_name="exercise4", workspace="juliusmo")

# ---------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------
data = dlipr.cifar.load_cifar10()

# plot some example images
dlipr.utils.plot_examples(data, fname='examples.png')

print(data.train_images.shape)
print(data.train_labels.shape)
print(data.test_images.shape)
print(data.test_labels.shape)

# preprocess the data in a suitable way
X_train = data.train_images
X_test = data.test_images

print('the shape', X_train.shape)
print('the shape2', X_train.shape[1:])
'''
# convert integer RGB values (0-255) to float values (0-1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# convert class labels to one-hot encodings
Y_train = tf.keras.utils.to_categorical(data.train_labels, 10)
Y_test = tf.keras.utils.to_categorical(data.test_labels, 10)


# ----------------------------------------------------------
# Model and training
# ----------------------------------------------------------
# fancy-pancy: it started with a feeling.. next, I wrote down this CNN from scratch! Seriously!

model = models.Sequential([
    layers.Conv2D(32, 
                    kernel_size=(3, 3), 
                    strides=(1, 1),
                    activation='relu', 
                    padding='same', 
                    dilation_rate=1,
                    input_shape=X_train.shape[1:]),
    layers.Conv2D(32, 
                    kernel_size=(3, 3), 
                    strides=(1, 1),
                    activation='relu', 
                    padding='same', 
                    dilation_rate=1,
                    input_shape=X_train.shape[1:]),
    layers.MaxPooling2D(pool_size=(2, 2), 
                    strides=None, 
                    padding='same'),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, 
                    kernel_size=(3, 3), 
                    strides=(1, 1),
                    activation='relu', 
                    padding='same', 
                    dilation_rate=1,
                    input_shape=X_train.shape[1:]),
    layers.Conv2D(64, 
                    kernel_size=(3, 3), 
                    strides=(1, 1),
                    activation='relu', 
                    padding='same', 
                    dilation_rate=1,
                    input_shape=X_train.shape[1:]),
    layers.MaxPooling2D(pool_size=(2, 2), 
                    strides=None, 
                    padding='same'),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')])
    
    
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.SGD(lr=0.1),
    metrics=['accuracy'])


fit = model.fit(X_train, Y_train,
    batch_size=100,
    epochs=10,
    verbose=2,
    validation_split=0.1,
    callbacks=[])

print('model.summary():', model.summary())
# ----------------------------------------------------------
# Visualize data
# ----------------------------------------------------------
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

folder = 'results/'

if not os.path.exists(folder):
    os.makedirs(folder)

model.save(folder + 'my_model')

# Confusion matrix and classification image examples 
Yp = model.predict(X_test)
yp = np.argmax(Yp, axis=1)

for i in range(20):

    dlipr.utils.plot_prediction(
        Yp[i],
        data.test_images[i],
        data.test_labels[i],
        data.classes,
        fname=folder + 'test-%i.png' % i)

fig_conv = dlipr.utils.plot_confusion(yp, data.test_labels, data.classes,
                                 fname=folder + 'confusion.png')
experiment.log_figure(figure=fig_conv)

    
# loss and accuray of test and validation, stored in folder
n_epochs = len(fit.history['loss'])  # due to earlystopping callbacks

fig_loss_acc, (ax1, ax2) = plt.subplots(2)

ax1.plot(np.arange(1,n_epochs+1), fit.history['loss'], label="training")
ax1.plot(np.arange(1,n_epochs+1), fit.history['val_loss'], label="validation")
ax1.grid(True)
plt.close(fig_loss_acc)
ax1.set(ylabel="loss")
ax1.legend()
ax2.plot(np.arange(1,n_epochs+1), fit.history['accuracy'], label="training")
ax2.plot(np.arange(1,n_epochs+1), fit.history['val_accuracy'], label="validation")
ax2.grid(True)
ax2.set(xlabel="epochs", ylabel="accuracy")
ax2.legend()
fig_loss_acc.savefig(folder + 'loss_accuracy.png', bbox_inches='tight')
'''
