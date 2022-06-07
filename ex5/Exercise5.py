from comet_ml import Experiment
import dlipr
from tensorflow import keras
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
models = keras.models
layers = keras.layers

# Set up YOUR experiment - login to comet, create new project (for new exercise)
# and copy the statet command
# or just change the name of the workspace, and the API (you can find it in the settings)
#experiment = Experiment(api_key="jQH3I2Zcq46PU16iZDsMtfYIS",
#                        project_name="exercise5", workspace="juliusmo")

"""
Exercise 5:

Task 1: Classify the magnetic phases in terms of
- a fully connected layer (FCL)
- a convolutional neural network (CNN)
- the toy model (s. lecture slides)
Plot test accuracy vs. temperature for both networks and for the toy model.

Task 2: Discriminative localization
Pick out two correctly and two wrongly classified images from the CNN.
Look at Exercise 4, task 2 (visualize.py) to extract weights and feature maps from the trained model.
Calculate and plot the class activation maps and compare them with the images in order to see which regions lead to the class decision.

Hand in a printout of your commented code and plots.

If you are interested in the data generation look at MonteCarlo.py.
"""


folder = 'results/'

if not os.path.exists(folder):
    os.makedirs(folder)

# Load the Ising dataset
data = dlipr.ising.load_data()

# plot some examples
data.plot_examples(5, fname= folder + 'examples.png')

# features: images of spin configurations
X_train = data.train_images
X_test = data.test_images

# classes: simulated temperatures
T = data.classes

# labels: class index of simulated temperature
# create binary training labels: T > Tc?
Tc = 2.27

y_train = T[data.train_labels] > Tc
y_test = T[data.test_labels] > Tc

# one-hot encodings
Y_train = keras.utils.to_categorical(y_train, 2)
Y_test = keras.utils.to_categorical(y_test, 2)

# transforming tarining and test data for FC and CNN input
X_train_FC = X_train.reshape(-1, 32**2)
X_test_FC = X_test.reshape(-1, 32**2)

X_train_CNN = X_train[:,:,:,np.newaxis]
X_test_CNN = X_test[:,:,:,np.newaxis]

'''
####################### Task 1.1b) ##################################
#####################################################################

# transforming training and test data into vector for FC network
#X_train_FC = X_train.reshape(-1, 32**2)
#X_test_FC = X_test.reshape(-1, 32**2)

#################################
# setting up FC network
nodes = 100

model = models.Sequential([
    layers.Dense(nodes, input_shape=(32**2,)),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(2),
    layers.Activation('softmax')])

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(lr=1e-3),
    metrics=['accuracy'])

before = model.evaluate(X_train_FC, Y_train)
print('loss before training:', before[0])
print('accuracy before training:', before[1])

# training network for 10 epochs
fit = model.fit(
    X_train_FC, Y_train,
    epochs=10,
    verbose=2,
    validation_split=0.1,  # split off 10% of training data for validation
    callbacks=[keras.callbacks.CSVLogger(folder + 'history.csv')])

print('model.summary()_FC', model.summary())

model.save(folder + 'model_FC')
#################################   
    
# plotting loss and accuracy during training
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
fig_loss_acc.savefig(folder + 'loss_accuracy_FC.png', bbox_inches='tight')

#################################
# test accuracy

acc_test = model.evaluate(X_test_FC, Y_test, verbose=0)
print('Test accuracy for FC:', acc_test[1])
#Test accuracy for FC: 0.9745
'''

'''
####################### Task 1.1c) ##################################
####################################################################

# transforming training and test data
#X_train_CNN = X_train[:,:,:,np.newaxis]
#X_test_CNN = X_test[:,:,:,np.newaxis]


# setting up the CNN model
model = models.Sequential([
    layers.Conv2D(12, 
                    kernel_size=(3, 3), 
                    strides=(1, 1),
                    activation='relu', 
                    padding='same', 
                    dilation_rate=1,
                    input_shape=X_train_CNN.shape[1:]),
    layers.Conv2D(12, 
                    kernel_size=(3, 3), 
                    strides=(1, 1),
                    activation='relu', 
                    padding='same', 
                    dilation_rate=1),
    layers.MaxPooling2D(pool_size=(2, 2), 
                    strides=None, 
                    padding='same'),
    layers.Dropout(0.25),
    layers.Conv2D(24, 
                    kernel_size=(3, 3), 
                    strides=(1, 1),
                    activation='relu', 
                    padding='same', 
                    dilation_rate=1),
                    
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dense(2, activation='softmax')])
    
    
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.SGD(lr=0.1),
    metrics=['accuracy'])


fit = model.fit(X_train_CNN, Y_train,
    batch_size=100,
    epochs=10,
    verbose=2,
    validation_split=0.1,
    callbacks=[])

model.save(folder + 'model_CNN')

print('model.summary()_CNN:', model.summary())

#################################   
    
# plotting loss and accuracy during training
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
fig_loss_acc.savefig(folder + 'loss_accuracy_CNN.png', bbox_inches='tight')

#################################
# test accuracy CNN

acc_test = model.evaluate(X_test_CNN, Y_test, verbose=0)
print('Test accuracy for CNN:', acc_test[1])
#Test accuracy for CNN: 0.98625
'''

'''
####################### Task 1.2 ##################################
#####################################################################

model_FC = keras.models.load_model('results/model_FC')
model_CNN = keras.models.load_model('results/model_CNN')

temperatures = np.arange(1, 3.51, 0.1)

acc_FC = []
acc_CNN = []
for temp in temperatures:
        indices = T[data.test_labels] == temp
        indices = np.where(indices == 1)
        indices = indices[0]
        X_temp_FC = X_test_FC[indices]
        X_temp_CNN = X_test_CNN[indices]
        Y_temp = Y_test[indices]
        score_FC = model_FC.evaluate(X_temp_FC, Y_temp, verbose=0)
        score_CNN = model_CNN.evaluate(X_temp_CNN, Y_temp, verbose=0)
        acc_FC.append(score_FC[1])
        acc_CNN.append(score_CNN[1])

acc_FC = np.array(acc_FC)
acc_CNN = np.array(acc_CNN)
print('acc_FC', acc_FC)
print('acc_CNN', acc_CNN)

plt.figure(1)
plt.plot(temperatures, acc_FC, 'b', label='FC')
plt.plot(temperatures, acc_FC, 'b.')
plt.plot(temperatures, acc_CNN, 'r', label='CNN')
plt.plot(temperatures, acc_CNN, 'r.')
plt.vlines(Tc, ymin=0, ymax=1, label='Tc')
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('T', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.grid()

plt.savefig(folder + 'accuracy_T.png', bbox_inches='tight')

'''
####################### Task 2.1 ####################################
#####################################################################

model_CNN = keras.models.load_model('results/model_CNN')

# selecting images for plotting close to Tc
temperatures = np.arange(1, 3.51, 0.1)
#print('temperatures[14]', temperatures[13])  

indices = T[data.test_labels] == temperatures[13]
indices = np.where(indices == 1)
indices = indices[0]
X_temp = X_test_CNN[indices]
Y_temp = Y_test[indices]

# prediction of model_CNN
prediction = model_CNN.predict(X_temp)

#X_temp = np.squeeze(X_temp)

'''
# plotting
for i in range(30):

    dlipr.utils.plot_prediction(
        prediction[i],
        X_temp[i],
        np.argmax(Y_temp[i]),
        np.array([1,0]),
        fname=folder + 'testimages/image{0}.png'.format(i))

'''

####################### Task 2.2 ############################
#####################################################################

###############################
# filters

#mylayers = model_CNN.layers
#for i,layer in enumerate(mylayers):
#    print('layer{0}'.format(i), layer)
    
# getting weights from last conv layer and plotting filters
weights, biases = model_CNN.layers[4].get_weights()
print('shape.weights:', np.shape(weights))

# normalize weights for plotting
weights_min = np.min(weights)
weights_plot = (weights - weights_min) / np.max(weights - weights_min)

# plotting 24 filters of first 3 channels of last conv layer
Nfilter = 24
n1 = 6
n2 = 4

plt.figure(2)
for i in range(Nfilter):
	filters = weights_plot[:, :, 0:3, i]
	#print('shape filters:', np.shape(filters))
	plt.subplot(n1,n2,i+1)
	plt.axis("off")
	plt.imshow(filters)
	
plt.suptitle('Filters 3rd conv-layer', va='bottom')	
plt.savefig('results/filters_conv3.png', bbox_inches='tight')	

###############################

# feature maps
index = [3, 24, 7, 28]

def visualize_activation(A, name='conv'):
    nx, ny, nf = A.shape
    n = np.ceil(nf**.5).astype(int)
    fig, axes = plt.subplots(n, n, figsize=(5, 5))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0, wspace=0)
    for i in range(n**2):
        ax = axes.flat[i]
        if i < nf:
            ax.imshow(A[..., i], origin='upper', cmap=plt.cm.Greys)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.axis('Off')
    fig.suptitle( name, va='bottom')
    fig.savefig('results/' + '%s.png' % name, bbox_inches='tight')
    return fig
    
print('model.layers:', model_CNN.layers)
print('len', len(model_CNN.layers))
print('type model.layers:', type(model_CNN.layers[0]))
print('type model.layers:', type(model_CNN.layers[1]))
print('type model.layers:', type(model_CNN.layers[4]))

#conv_layers = [l for l in model.layers if type(l) == layers.Conv2D]
conv_layers = [model_CNN.layers[4]]
#print('len(conv_layers)', len(conv_layers))
for index in index:
    for j, conv in enumerate(conv_layers):
        conv_model = keras.models.Model(model_CNN.inputs, [conv.output])
        # plot the activations for image with index
        #Xin = X_test[i][np.newaxis]
        Xin = X_temp[index][np.newaxis]
        Xout1 = conv_model.predict(Xin)[0]
        fig = visualize_activation(Xout1, 'image{0}_conv3_24featuremaps'.format(index))


#################################
#################################

'''
# Load the Ising dataset
data = dlipr.ising.load_data()

# plot some examples
data.plot_examples(5, fname= folder + 'examples.png')

# features: images of spin configurations
X_train = data.train_images
X_test = data.test_images

# classes: simulated temperatures
T = data.classes

# labels: class index of simulated temperature
# create binary training labels: T > Tc?
Tc = 2.27

y_train = T[data.train_labels] > Tc
y_test = T[data.test_labels] > Tc

# one-hot encodings
Y_train = keras.utils.to_categorical(y_train, 2)
Y_test = keras.utils.to_categorical(y_test, 2)

# transforming tarining and test data for FC and CNN input
X_train_FC = X_train.reshape(-1, 32**2)
X_test_FC = X_test.reshape(-1, 32**2)

X_train_CNN = X_train[:,:,:,np.newaxis]
X_test_CNN = X_test[:,:,:,np.newaxis]
'''
