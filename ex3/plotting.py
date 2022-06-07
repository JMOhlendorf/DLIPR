import matplotlib.pyplot as plt
import numpy as np
import os


n_trial_max = 15
n_sub = 4

data = np.load('results/' +'hyper_val_array_ntrial_max={0}_nsub={1}_new.npz'.format(n_trial_max,n_sub),
                                                                             allow_pickle=True)



val_loss_arr = data['val_loss']
hyperpar_arr = data['hyperpar']
print(val_loss_arr)
val_loss_best = np.zeros((n_trial_max,n_sub))



for i in range(n_trial_max):
    for j in range(n_sub):
        val_loss_best[i,j] = np.min(val_loss_arr[i,j])
    
    

best_index = np.unravel_index(np.argmin(val_loss_best, axis=None), val_loss_best.shape)
print('index:', best_index)
print('min:', np.min(val_loss_best))
print(val_loss_arr)
print('best_list:', val_loss_arr[best_index[0], best_index[1]])
print(hyperpar_arr)
print('best_list:', hyperpar_arr[best_index[0], best_index[1]])


#print('min_test:', val

    
#print(val_loss_arr)
##print('mean:', np.mean(val_loss_arr, axis=1))
#hyperpar_arr = data['hyperpar']
#print(hyperpar_arr)


plt.figure(0)
plt.errorbar(np.arange(1,n_trial_max+1,1), np.mean(val_loss_best, axis=1),
             yerr=np.std(val_loss_best, axis=1), fmt='-o', capsize=5)
plt.xticks(np.arange(1,n_trial_max+1,1))
plt.grid()
plt.xlabel('#trials', fontsize=15)
plt.ylabel('validation loss', fontsize=15)
plt.savefig('results/' + 'ntrialmax{0}_nsub{1}_new.png'.format(n_trial_max, n_sub))
#plt.savefig('ntrialmax{0}_nsub{1}.png'.format(n_trial_max, n_sub))
plt.show()






