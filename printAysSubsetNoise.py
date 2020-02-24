import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

thsd = np.load('../criticalData/thsd_uncertain.npz')['arr_0']

# Cel, CelNoise20, CelNoise40, CelNoise60
# Diri, DiriNoise20, DiriNoise40, DiriNoise60 
modelTypeList = ['Cel', 'CelNoise20', 'CelNoise40', 'CelNoise60',
                              'Diri', 'DiriNoise20', 'DiriNoise40', 'DiriNoise60']

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
