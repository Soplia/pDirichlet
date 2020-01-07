import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

npzfile = np.load('../data/testM.npz')
outputs = npzfile['arr_0']
predictions = npzfile['arr_1']

numClass = 10
beliefRaw = outputs
evidence = beliefRaw * (beliefRaw > 0)
alpha = evidence + 1
###### Chose the belief you like ####
belief1 = beliefRaw
belief2 = evidence
belief3 = (evidence / np.sum(alpha, axis= 1, keepdims = True))
# Sa = np.sum(alpha, axis=1)
# Sa = np.expand_dims(Sa, axis=0)
# Sa = np.repeat(Sa, alpha.shape[1], axis = 0)
# # (8400, 10)
# Sa = np.transpose(Sa)
# belief4 = evidence / Sa
belief = belief4
######################################


probility = (alpha / np.sum(alpha, axis= 1, keepdims = True))
uncerty = (numClass / np.sum(alpha, axis= 1, keepdims = True))

beliefSorted = np.empty((1, 10))
probilitySorted = np.empty((1, 10))
for i in range(belief.shape[0]):
    onebelief = belief[i, :]
    oneprobility = probility[i, :]
    ibs = np.flip(np.argsort(belief[i, :]))
    beliefSorted = np.vstack((beliefSorted, onebelief[ibs]))
    probilitySorted = np.vstack((probilitySorted, oneprobility[ibs]))
beliefSorted = beliefSorted[1:, :]
probilitySorted = probilitySorted[1:, :]

beliefCumsum = np.cumsum(beliefSorted, axis= 1)
probilityCumsum = np.cumsum(probilitySorted, axis= 1)

threshold = np.linspace(start= 0, stop= 1, endpoint= True, num= 10)
resultbelief = []
resultprobility = []
numClassMatirx = np.full((alpha.shape[0], 1), numClass)
for th in threshold:
    thMatirx = np.full(alpha.shape, th)
    maskbelief = beliefCumsum >= thMatirx
    maskprobility = probilityCumsum >= thMatirx
    numCandinatorBelief = numClassMatirx - np.sum(maskbelief, axis= 1) + 1
    meanCandinatorBelief = np.mean(numCandinatorBelief)
    resultbelief.append(meanCandinatorBelief)
    #resultbelief.append(np.mean(numClassMatirx - np.sum(maskbelief, axis= 1) + 1))
    resultprobility.append(np.mean(numClassMatirx - np.sum(maskprobility, axis= 1) + 1))

fig, axe = plt.subplots()
#axe.plot(threshold, resultbelieflief)
axe.plot(threshold, resultbelief, color= 'r', marker= '>', label= 'belief')
axe.set_xlabel('Values of threshold')
axe.set_ylabel('Num of candinator for Belief')
axe.legend(loc= 2)

axe1 = axe.twinx()
#axe1.plot(threshold, resultprobility)
axe1.plot(threshold, resultprobility, color= 'k', marker= 'o', label= 'probility')
axe1.set_ylabel('Num of candinator for Probility')
axe1.legend(loc= 4)

plt.show()
