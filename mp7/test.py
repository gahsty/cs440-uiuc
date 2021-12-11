import numpy as np
t = np.load('C:/Users/gahty/Desktop/UIUC/cs440/mp7/checkpoint.npy')
N_t = np.load('C:/Users/gahty/Desktop/UIUC/cs440/mp7/checkpoint_N.npy')

ta = np.load('C:/Users/gahty/Desktop/UIUC/cs440/mp7/data/checkpoint1.npy')
ta_N = np.load('C:/Users/gahty/Desktop/UIUC/cs440/mp7/data/checkpoint1_N.npy')

diff = t==ta
are_same = np.all(diff)
indx_diff = np.where(diff == False)


print(indx_diff)