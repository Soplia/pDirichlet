import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# Load data from test dataset
npzfile = np.load('../data/testDiri.npz')
#npzfile = np.load('../data/testCel.npz')
#npzfile = np.load('../data/testNoise20.npz')
outputs = npzfile['arr_0']
labels = npzfile['arr_1']
# # Load data from roated digit
# outputs = np.load('../data/roatedDigitOutput.npz')['arr_0']
# label = np.load('../data/roatedDigitLabel.npz')['arr_0']
# predictions = np.full((outputs.shape[0], 1), label)

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

np.savez('../data/thsd_uncertain', thsd_uncertain)
np.savez('../data/aysDiri', np.array(resultbelief), np.array(acc))
#np.savez('../data/aysCel.np', np.array(resultbelief), np.array(resultprobility))
#np.savez('../data/aysNoise20.np', np.array(resultbelief), np.array(resultprobility))
print ("Finish saving files!!")

