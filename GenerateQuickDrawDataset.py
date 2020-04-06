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
#import input_data

idx = np.linspace(0, 120535, num= 1000, dtype= np.int32)

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

feaNp = np.concatenate([clocknp[idx], cloudnp[idx], cupnp[idx], eyeglassesnp[idx],
                  laddernp[idx], pantsnp[idx], scissorsnp[idx], sunnp[idx], 
                  tablenp[idx], umbrellanp[idx]], axis= 0)

label = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
tarNp = np.repeat(label, 1000)

np.savez('../data/quickdrawdataset.npz', feaNp, tarNp)
print ('Finish Saving QuickDrawDataset')


