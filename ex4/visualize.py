from comet_ml import Experiment
import numpy as np
import matplotlib.pyplot as plt
import dlipr
from tensorflow import keras
layers = keras.layers


# Set up YOUR experiment - login to comet, create new project (for new exercise)
# and copy the statet command
# or just change the name of the workspace, and the API (you can find it in the settings)
experiment = Experiment(api_key="jQH3I2Zcq46PU16iZDsMtfYIS",
                        project_name="exercise4", workspace="juliusmo")

# ----------------------------------------------------------
# Load your model
# ----------------------------------------------------------
model = keras.models.load_model('results/my_model')

print('model.summary():', model.summary())
print('model.layers:', model.layers)

# Note: You need to pick the right convolutional layers from your network here
conv1 = model.layers[1]
conv2 = model.layers[3]


# ----------------------------------------------------------
# Plot the convolutional filters in the first layer
# ----------------------------------------------------------
weights, biases = model.layers[0].get_weights()

print('shape.weights:', np.shape(weights))

plt.figure(0)
plt.hist(weights[:, :, :, 1].flatten())
plt.savefig('histme.png', bbox_inches='tight')
plt.show()

# normalize weights
weights_min = np.min(weights)
weights_max = np.max(weights)
weights = (weights - weights_min) / np.max(weights - np.min(weights))
#weights = (weights - weights_min) / (weights_max - weights_min)
#weights[:,:,2] = 0.99
#weights[:,:,2] = 0.01
#weights[:,:,2] = 0.5
plt.figure(1)
plt.hist(weights[:, :, :, 1].flatten())
plt.savefig('histme_after.png', bbox_inches='tight')
plt.show()


# plotting all 32 filters of the first layer
Nfilter = 32
n1 = 8
n2 = 4

plt.figure(2)
#plt.title('Filters 1st layer')
for i in range(Nfilter):
	filters = weights[:, :, :, i]
	print('shape filters:', np.shape(filters))
	plt.subplot(n1,n2,i+1)
	plt.axis("off")
	plt.imshow(filters)
	
plt.suptitle('myname', va='bottom')	
plt.savefig('filters_finaltestit.png', bbox_inches='tight')	


'''
# ---------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------
data = dlipr.cifar.load_cifar10()

# prepare the test set the same way as in your training
X_test = data.test_images
X_test = X_test.astype('float32') / 255


i = 12  # choose a good test sample


# ----------------------------------------------------------
# Plot the picture with predictions
# ----------------------------------------------------------
# Confusion matrix and classification image examples 
Yp = model.predict(X_test)

dlipr.utils.plot_prediction(
        Yp[i],
        data.test_images[i],
        data.test_labels[i],
        data.classes,
        fname= 'results/Test-%i.png' % i)


# ----------------------------------------------------------
# Plot activations in convolution layers
# ----------------------------------------------------------
def visualize_activation(A, name='conv'):
    print('I entered the function')
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
    fig.suptitle(name, va='bottom')
    #fig.savefig('Hans.png', bbox_inches='tight')
    fig.savefig('%s.png' % name, bbox_inches='tight')
    return fig

print('type model.layers:', type(model.layers[0]))
print('type model.layers:', type(model.layers[1]))
print('type model.layers:', type(model.layers[2]))

#conv_layers = [l for l in model.layers if type(l) == layers.Conv2D]
conv_layers = [(model.layers[0]), (model.layers[4])]
print('len(conv_layers)', len(conv_layers))

for j, conv in enumerate(conv_layers):
    print('In the loop')

    conv_model = keras.models.Model(model.inputs, [conv.output])
    # plot the activations for test sample i
    Xin = X_test[i][np.newaxis]
    Xout1 = conv_model.predict(Xin)[0]
    fig = visualize_activation(Xout1, 'results/image%i-conv%i' % (i, j))
    experiment.log_figure(figure=fig)
    
'''
