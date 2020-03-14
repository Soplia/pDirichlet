import torch
import torch.nn.functional as F
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Cel20 , Diri20
modelType = 'Diri20' 

npzfile = np.load('../data/test{0}.npz'.format(modelType))
outputs =npzfile['arr_0']
trueLabels = npzfile['arr_1']
modelPredictions = np.argmax(outputs, axis= 1)

numClass = 10
beliefRaw = outputs
evidence = beliefRaw * (beliefRaw > 0)
alpha = evidence + 1
belief = alpha
######################################

print ("AysSubsetPR...")

beliefSorted = np.empty((1, 10))
idxBeliefSorted = np.empty((1, 10))
for i in range(belief.shape[0]):
    oneBelief = belief[i, :]
    ibs = np.flip(np.argsort(belief[i, :]))
    beliefSorted = np.vstack((beliefSorted, oneBelief[ibs]))
    idxBeliefSorted = np.vstack((idxBeliefSorted, ibs))
beliefSorted = beliefSorted[1:, :]
idxBeliefSorted = idxBeliefSorted[1:, :]

beliefCumsum = np.cumsum(beliefSorted, axis= 1)
K = np.ones_like(beliefCumsum)
K = np.cumsum(K, axis= 1)
uncertain_subset = K / beliefCumsum

thsd_uncertain = np.arange(start= 1, stop= 0, step= -0.05, dtype= np.float)

recall = []
precision = []
numClassMatirx = np.full((alpha.shape[0], 1), numClass)
for th in thsd_uncertain:
    thMatirx = np.full(alpha.shape, th)
    maskbelief = uncertain_subset >= thMatirx
    subsetPredictions = []
    for i in range(idxBeliefSorted.shape[0]):  
        predictions = []
        for j in range(numClass):
            if maskbelief[i, j] == False:
                predictions.append(idxBeliefSorted[i, j].astype(np.int))
            else:
                predictions.append(idxBeliefSorted[i, j].astype(np.int))
                break;
        if trueLabels[i] in predictions:
            subsetPredictions.append(trueLabels[i])
        else:
            subsetPredictions.append(-1)
    precision.append(precision_score(trueLabels, subsetPredictions, average='micro'))
    recall.append(recall_score(trueLabels, subsetPredictions, average='micro'))
    
np.savez('../data/aysSubsetPR{}'.format(modelType), np.array(precision), np.array(recall))
print ("Finish saving files!!")
