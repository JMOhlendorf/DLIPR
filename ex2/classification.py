"""
---------------------------------------------------
Exercise 2 - Classification
---------------------------------------------------
Suppose we want to classify some data (4 samples) into 3 distinct classes: 0, 1, and 2.
We have set up a network with a pre-activation output z in the last layer.
Applying softmax will give the final model output.
input X ---> some network --> z --> y_model = softmax(z)

We quantify the agreement between truth (y) and model using categorical cross-entropy.
J = - sum_i (y_i * log(y_model(x_i))

In the following you are to implement softmax and categorical cross-entropy
and evaluate them values given the values for z.
"""
from __future__ import print_function
from comet_ml import Experiment
import numpy as np
import tensorflow as tf


experiment = Experiment(api_key="jQH3I2Zcq46PU16iZDsMtfYIS",
                        project_name="exercise2", workspace="juliusmo")

num_classes = 3
# Data: 4 samples with the following class labels (input features X irrelevant here)
y_cl = np.array([0, 0, 2, 1]) #(4,)
print('y_cl.shape:', y_cl.shape)

# output of the last network layer before applying softmax
z = np.array([
    [4,    5,    1],
    [-1,  -2,   -3],
    [0.1, 0.2, 0.3],
    [-1,  17,    1]
    ]).astype(np.float32) #(4,3)

print('z.shape:', z.shape)
# TensorFlow implementation as reference. Make sure you get the same results!
print('\nTensorFlow 2.0 ------------------------------ ')


def crossentropy(x, y):
    return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(x), axis=1))


y = tf.one_hot(y_cl, 3)
y_model = tf.nn.softmax(z)

print('(y:(TF)', y)
print('softmax(z)(TF)', y_model)
print('cross entropy(TF)', crossentropy(y_model, y))


print('\nMy solution ------------------------------ ')
# 1) Write a function that turns any class labels y_cl into one-hot encodings y.
#    0 --> (1, 0, 0)
#    1 --> (0, 1, 0)
#    2 --> (0, 0, 1)
#    Make sure that np.shape(y) = (4, 3) for np.shape(y_cl) = (4).


def to_onehot(y_cl, num_classes):
    y = np.zeros((len(y_cl), num_classes))
    for i,class_score in enumerate(y_cl):
        y[i, class_score] = 1
    return y

print('(y:(tensorflow)', y)
y_one = to_onehot(y_cl, 3)
print('y_cl:', y_cl)
print('y_one:', y_one)


# 2) Write a function that returns the softmax of the input z along the last axis.
def softmax(z):
    z_soft = np.zeros(z.shape)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z_soft[i,j] = np.exp(z[i,j])/ np.sum(np.exp(z[i]))
    return(z_soft)

z_soft = softmax(z)

print('softmax(z)(tensorflow)', y_model)
print('z_soft:', z_soft)

# 3) Compute the categorical cross-entropy between data and model.

def cross_entro(y_model, y_one):
    J_temp = np.multiply(y_one, np.log(y_model))
    J = -1/np.count_nonzero(J_temp)*np.sum(J_temp)
    return(J)
    
print('cross entropy(tensorflow)', crossentropy(y_model, y))
print('cross entropy:', cross_entro(z_soft, y_one))

# 4) Which classes are predicted by the model (maximum entry).
prediction = z_soft.argmax(axis=1)
print('Model prediction:', prediction)
print('Ground truth:', y_cl)


# 5) How many samples are correctly classified (accuracy)?
equal = np.equal(prediction, y_cl)
accuracy = np.sum(equal)/ len(equal)
print('Accuracy:', accuracy)
