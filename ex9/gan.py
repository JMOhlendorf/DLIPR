import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
layers = keras.layers
reg = keras.regularizers


def plot_images(images, figsize=(10, 10), fname=None):
    """ Plot some images """
    n_examples = len(images)
    dim = np.ceil(np.sqrt(n_examples))
    fig = plt.figure(figsize=figsize)

    for i in range(n_examples):
        plt.subplot(dim, dim, i + 1)
        img = np.squeeze(images[i])
        plt.imshow(img, cmap=plt.cm.Greys)
        plt.axis('off')
    plt.tight_layout()

    if fname is not None:
        fig.savefig(fname)

    return fig


def make_trainable(model, trainable):
    """ Helper to freeze / unfreeze a model """
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable


def generator_model(inp, n_channels=200):
    """ Generator network """
    z = layers.Dense(14 * 14 * n_channels, activity_regularizer=reg.l1_l2(1e-5))(inp)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)
    z = layers.Reshape([14, 14, n_channels])(z)
    z = layers.UpSampling2D(size=(2, 2))(z)
    z = layers.Conv2D(n_channels // 2, (3, 3), padding='same')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)
    z = layers.Conv2D(n_channels // 4, (3, 3), padding='same')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)
    return layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(z)


def discriminator_model(inp, drop_rate=0.25):
    """ Discriminator network """
    z = layers.Conv2D(256, (5, 5), padding='same', strides=(2, 2), activation='relu')(inp)
    z = layers.LeakyReLU(0.2)(z)
    z = layers.Dropout(drop_rate)(z)
    z = layers.Conv2D(512, (5, 5), padding='same', strides=(2, 2), activation='relu')(z)
    z = layers.LeakyReLU(0.2)(z)
    z = layers.Dropout(drop_rate)(z)
    z = layers.Flatten()(z)
    z = layers.Dense(256, activity_regularizer=reg.l1_l2(1e-5))(z)
    z = layers.LeakyReLU(0.2)(z)
    z = layers.Dropout(drop_rate)(z)
    return layers.Dense(2, activation='softmax')(z)
