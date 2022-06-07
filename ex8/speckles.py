import numpy as np
import matplotlib.pyplot as plt
import h5py

def load_data():
    filestruct=h5py.File('/net/scratch/deeplearning/AutoEncoder/complete_with_noise.mat', "r")

    data_normal=filestruct['data']['normal_images']
    data_speckle=filestruct['data']['speckle_images']

    data_normal=np.swapaxes(data_normal,0,1)
    data_speckle=np.swapaxes(data_speckle,0,1)
    
    x_train=data_normal[0:20000]
    x_test=data_normal[20000:-1]
    x_train_noisy=data_speckle[0:20000]
    x_test_noisy=data_speckle[20000:-1]
    
    return (x_train_noisy, x_train), (x_test_noisy, x_test)
    
def preprocess_data(data):

    data=np.log10(data + 0.01)
    max_val=np.max(data, axis=1)
    data=data/max_val.reshape(len(max_val), 1)
    data = np.reshape(data, (len(data), 64, 64, 1))
    data = np.clip(data, 0., 1.)
    
    return data

def plot_examples(x_noisy, x, fname='examples.png'):
    
    n = 10
    fig = plt.figure(figsize=(20, 4))
    for i in range(n):
        plt.subplot(2, n, i+1)
        plt.imshow(x_noisy[i,:,:,0])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, n, n+i+1)
        plt.imshow(x[i,:,:,0])
        plt.xticks([])
        plt.yticks([])

    fig.savefig(fname)
