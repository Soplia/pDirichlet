"""
It generates and trains a CNN model with loss function eq5
Saving the state_dict as a file 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
#import input_data

"""
It generates and trains a CNN model with loss function eq5
Saving the state_dict as a file 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

clocknp = np.load('../quickdrawdata/clock.npy')
cloudnp = np.load('../quickdrawdata/clock.npy')
cupnp = np.load('../quickdrawdata/clock.npy')
eyeglassesnp = np.load('../quickdrawdata/clock.npy')
laddernp = np.load('../quickdrawdata/clock.npy')
pantsnp = np.load('../quickdrawdata/clock.npy')
scissorsnp = np.load('../quickdrawdata/clock.npy')
sunnp = np.load('../quickdrawdata/clock.npy')
tablenp = np.load('../quickdrawdata/clock.npy')
umbrellanp = np.load('../quickdrawdata/clock.npy')

numOfSample = 1000
idx = np.linspace(0, 120535, num= numOfSample, dtype= np.int32)
label = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
tarNp = np.repeat(label, numOfSample)

feaNp = np.concatenate([clocknp[idx], cloudnp[idx], cupnp[idx], eyeglassesnp[idx],
                  laddernp[idx], pantsnp[idx], scissorsnp[idx], sunnp[idx], 
                  tablenp[idx], umbrellanp[idx]], axis= 0)

permutation = np.random.permutation(tarNp.shape[0])
shuffledFeaNp = feaNp[permutation, :]
shuffledTarNp = tarNp[permutation]

np.savez('../data/quickdrawdataset.npz', shuffledFeaNp, shuffledTarNp)
print ('Finish Saving QuickDrawDataset')


