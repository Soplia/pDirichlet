import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# Cel20, Cel20Noise1, Cel20Noise10, Cel20Noise20
# Diri20, Diri20Noise1, Diri20Noise10, Diri20Noise20 
modelType = 'Cel20' 
npzfile = np.load('../data/test{0}.npz'.format(modelType))

outputs = npzfile['arr_0']
labels = npzfile['arr_1']

# # Load data from roated digit
#outputs = np.load('../data/roatedDigitOutput.npz')['arr_0']
#label = np.load('../data/roatedDigitLabel.npz')['arr_0'].item()
#labels = np.zeros((outputs.shape[0], 1))
#labels.fill(label)

numClass = 10
beliefRaw = outputs
evidence = beliefRaw * (beliefRaw > 0)
alpha = evidence + 1
###### Chose the belief you like ####
belief1 = beliefRaw
belief2 = evidence
belief3 = (evidence / np.sum(alpha, axis= 1, keepdims = True))
belief4 = alpha
belief = belief4
######################################

print ("Ays...")

beliefSorted = np.empty((1, 10))
ibsSorted = np.empty((1, 10))
for i in range(belief.shape[0]):
    onebelief = belief[i, :]
    ibs = np.flip(np.argsort(belief[i, :]))
    beliefSorted = np.vstack((beliefSorted, onebelief[ibs]))
    ibsSorted = np.vstack((ibsSorted, ibs))
beliefSorted = beliefSorted[1:, :]
ibsSorted = ibsSorted[1:, :]

beliefCumsum = np.cumsum(beliefSorted, axis= 1)
K = np.ones_like(beliefCumsum)
K = np.cumsum(K, axis= 1)
uncertain_subset = K / beliefCumsum

thsd_uncertain = np.arange(start= 0, stop= .4, step= .005, dtype= np.float)

resultbelief = []
acc = []
numClassMatirx = np.full((alpha.shape[0], 1), numClass)

for th in thsd_uncertain:
    thMatirx = np.full(alpha.shape, th)
    maskbelief = uncertain_subset >= thMatirx

    numCandinatorBelief = numClassMatirx - np.sum(maskbelief, axis= 1) + 1
    meanCandinatorBelief = np.mean(numCandinatorBelief)
    resultbelief.append(meanCandinatorBelief) 
    
    cnt = 0
    for i in range(ibsSorted.shape[0]):  
        predictions = []
        for j in range(numClass):
            if maskbelief[i, j] == False:
                predictions.append(ibsSorted[i, j].astype(np.int))
            else:
                predictions.append(ibsSorted[i, j].astype(np.int))
                break;
        if labels[i] in predictions:
            cnt += 1
    
    acc.append(cnt / ibsSorted.shape[0])

#np.savez('../criticalData/thsd_uncertain', thsd_uncertain)
np.savez('../data/ays{}'.format(modelType), np.array(resultbelief), np.array(acc))
#np.savez('../data/aysRoatedNum', np.array(resultbelief), np.array(acc))

print ("Finish saving files!!")

