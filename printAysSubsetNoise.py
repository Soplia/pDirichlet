import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

thsd = np.load('../criticalData/thsd_uncertain.npz')['arr_0']

# Cel20, Cel20Noise20, Cel20Noise40, Cel20Noise60
# Diri20, Diri20Noise20, Diri20Noise40, Diri20Noise60 
modelTypeList = ['Cel20', 'Cel20Noise1', 'Cel20Noise10', 'Cel20Noise20',
                              'Diri20', 'Diri20Noise1', 'Diri20Noise10', 'Diri20Noise20']

beliefList = []
accList = []
thsdList = []
for modelType in modelTypeList:
    npzfile = np.load('../data/ays{}.npz'.format(modelType))
    beliefList.append(npzfile['arr_0'])
    accList.append(npzfile['arr_1'])
    thsdList.append(thsd)

marker=['s','^','*', '>'] * 4

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

for idx, belief in enumerate(beliefList):
    axes[0].plot(thsd, belief, marker= marker[idx], label= modelTypeList[idx])
axes[0].set_xlabel('uncertainty threshold')
axes[0].set_ylabel('number of candidates')
axes[0].legend(loc= 4)

for idx, acc in enumerate(accList):
    axes[1].plot(thsd, acc, marker= marker[idx], label= modelTypeList[idx])
axes[1].set_ylabel('accuracy')
axes[1].set_xlabel('uncertainty threshold')
axes[1].legend(loc= 4)

plt.show()
