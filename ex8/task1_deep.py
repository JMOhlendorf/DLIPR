from comet_ml import Experiment
import numpy as np
import dlipr
import speckles
from tensorflow import keras
models = keras.models
layers = keras.layers
backend = keras.backend

from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Input
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Set up YOUR experiment - login to comet, create new project (for new exercise)
# and copy the statet command
# or just change the name of the workspace, and the API (you can find it in the settings)
#experiment = Experiment(api_key="<HIDDEN>",
#                        project_name="exercise8", workspace="henry")


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


#defining the input layer
input = Input(shape=(64,64,1), name = 'input')
x = Conv2D(16, kernel_size = (3,3), padding='same', activation='relu', name = 'encoder_conv_1')(input)
x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
x = Conv2D(8, kernel_size = (3,3), padding='same', activation='relu', name = 'encoder_conv_2')(x)
x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
x = Conv2D(8, kernel_size = (3,3), padding='same', activation='relu', name = 'encoder_conv_3')(x)
x = MaxPooling2D(pool_size=(2,2), padding='same')(x)

x = Conv2D(8, kernel_size = (3,3), padding='same', activation='relu', name = 'decoder_conv_4')(x)
x = UpSampling2D(size=(2,2))(x)
x = Conv2D(8, kernel_size = (3,3), padding='same', activation='relu', name = 'decoder_conv_5')(x)
x = UpSampling2D(size=(2,2))(x)
x = Conv2D(8, kernel_size = (3,3), padding='same', activation='relu', name = 'decoder_conv_6')(x)
x = UpSampling2D(size=(2,2))(x)
output = Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='linear', name='reconstructed_output')(x)
'''
model = Model(inputs=input, outputs=output, name='CAE')
model.compile(optimizer='adam', loss='mse')
#loss='mse'
model.summary()

model.fit(x_train_noisy, x_train, batch_size=8, epochs=10, verbose=2)

model.save('model_mse.h5')

out_images = model.predict(x_test_noisy)
'''
model = models.load_model('model_mse.h5')  # load your trained model -> you only have to train the model once!
model.summary()
out_images = model.predict(x_test_noisy)
#PLOTTING THE DATA

n = 10 # Number of images to be displayed
plt.figure(figsize=(15,4))
for i in range(n):
    #display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(64, 64))
    
    #display reconstruction
    ax = plt.subplot(2, n, i + 1 +n)
    plt.imshow(out_images[i].reshape(64, 64))
plt.savefig('reconstruction_deep_autoencoder_mse_test.png')


speckles.plot_examples(x_test_noisy, x_test, fname='speckle_examples.png')
print(x_train_noisy.shape)

# some indices of interesting test data (it is not necessary to inspect all)
n = [1, 2, 4, 6, 10, 12, 14, 16, 22, 25, 27, 28, 30, 32, 35, 37, 39, 40, 41, 48, 49, 50, 52, 57, 61, 63, 64, 67, 70, 76]