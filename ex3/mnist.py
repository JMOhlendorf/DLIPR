from comet_ml import Experiment
import dlipr
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
models = keras.models
layers = keras.layers


# Set up YOUR experiment - login to comet, create new project (for new exercise)
# and copy the statet command
# or just change the name of the workspace, and the API (you can find it in the settings)
experiment = Experiment(api_key="jQH3I2Zcq46PU16iZDsMtfYIS",
                        project_name="exercise3", workspace="juliusmo")

# ----------------------------------------------
# Data
# ----------------------------------------------
# IF THE FOLLOWING LINE BREAKS - Check if you added already: software community/dlipr to your .profile as discussed in lecture 1
# after adding the line: login and logout WITHOUT restoring the old session!
data = dlipr.mnist.load_data()  

# plot some examples
data.plot_examples(fname='examples.png')

# reshape the image matrices to vectors
X_train = data.train_images.reshape(-1, 28**2)
X_test = data.test_images.reshape(-1, 28**2)
print('%i training samples' % X_train.shape[0])
print('%i test samples' % X_test.shape[0])

# convert integer RGB values (0-255) to float values (0-1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# convert class labels to one-hot encodings
Y_train = tf.keras.utils.to_categorical(data.train_labels, 10)
Y_test = tf.keras.utils.to_categorical(data.test_labels, 10)

# ----------------------------------------------
# Model and training
# ----------------------------------------------
###############################################################
# own code
###############################################################

n_trial_max = 15
n_sub = 4
val_loss_arr = np.empty((n_trial_max, n_sub), dtype=object)
hyperpar_arr = np.empty((n_trial_max, n_sub), dtype=object)

# make output directory
folder = 'results/'

if not os.path.exists(folder):
    os.makedirs(folder)
    
n_trial = 0

while n_trial < n_trial_max:
    print('------------n_trial:{0}---------------'.format(n_trial))
    for j in range(n_sub):
        hyperpar_templist = []
        val_loss_templist = []
        print('------------n_sub:{0}---------------'.format(j))
        for i in range(n_trial+1):
            lrat = np.random.choice([0.0001, 0.001, 0.01, 0.1, 1])
            mbatch = np.random.choice([10, 40, 70, 100, 130, 160])
            dropout = np.random.choice([0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
            actfunc = np.random.choice(['relu', 'tanh', 'sigmoid'])
            
            hyperpar_templist.append([lrat, mbatch, dropout, actfunc])
            
            model = models.Sequential([
                layers.Dense(128, input_shape=(784,)),
                layers.Activation(actfunc),
                layers.Dropout(dropout),
                layers.Dense(10),
                layers.Activation('softmax')])
            
            #print(model.summary())
            
            model.compile(
                loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=lrat),
                metrics=['accuracy'])
            
            results = model.fit(
                X_train, Y_train,
                batch_size=100,
                epochs=4,
                verbose=2,
                validation_split=0.1,  # split off 10% of training data for validation
                callbacks=[keras.callbacks.CSVLogger(folder + 'history.csv')])
                
            val_loss_templist.append(results.history['val_loss'][-1]) 
            
        hyperpar_arr[n_trial,j] = hyperpar_templist
        val_loss_arr[n_trial,j] = val_loss_templist
        
    n_trial += 1

np.savez(folder + 'hyper_val_array_ntrial_max={0}_nsub={1}_new.npz'.format(n_trial_max,n_sub), 
                                                                    hyperpar=hyperpar_arr,
                                                                    val_loss=val_loss_arr)
'''
plt.figure(0)
#plt.plot(np.arange(0,n_trial,1), np.mean(val_loss_arr, axis=1))
plt.errorbar(np.arange(0,n_trial,1), np.mean(val_loss_arr, axis=1), yerr=np.std(val_loss_arr, axis=1), fmt='-o')
plt.xticks(np.arange(0,n_trial,1))
plt.ylim(0,10)
plt.xlabel('#trial', fontsize=15)
plt.ylabel('validation loss', fontsize=15)
plt.savefig('results/' + 'ntrial{0}_nsub{1}.png'.format(n_trial, n_sub))
plt.show()


#print('Printing val_loss again')
#print(results.history['val_loss'])


# ----------------------------------------------
# Some plots
# ----------------------------------------------

# predicted probabilities for the test set
Yp = model.predict(X_test)
yp = np.argmax(Yp, axis=1)

# plot some test images along with the prediction
for i in range(20):

    dlipr.utils.plot_prediction(
        Yp[i],
        data.test_images[i],
        data.test_labels[i],
        data.classes,
        fname=folder + 'test-%i.png' % i)

# plot the confusion matrix
fig = dlipr.utils.plot_confusion(yp, data.test_labels, data.classes,
                                 fname=folder + 'confusion.png')
experiment.log_figure(figure=fig)
'''