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
#npzfile = np.load('../data/testCel20Noise20.npz')

outputs = torch.from_numpy(npzfile['arr_0'])
trueLabels = npzfile['arr_1']
modelPredictions = torch.argmax(outputs.data, dim= 1).numpy()

numClass = 10
beliefRaw = outputs
evidence = beliefRaw * (beliefRaw > 0)
alpha = evidence + 1
belief = alpha
######################################

print ("AysPR...")

S = torch.sum(belief, dim= 1)
uncertain = numClass / S
thsd_uncertain = np.arange(start= 0, stop= 1, step= 0.1, dtype= np.float)

recall = []
precision = []
for th in thsd_uncertain:
    thPredictions = []
    for i in range(uncertain.shape[0]):  
        if uncertain[i] < th:
            #thPredictions.append(modelPredictions[i])
            thPredictions.append(trueLabels[i])
        else:
            # if current uncertain is greater than the threshold, 
            # then reject this prediction by assign -1 to this sample label
            #thPredictions.append(-1)
            thPredictions.append(modelPredictions[i])

    precision.append(precision_score(trueLabels, thPredictions, average='weighted'))
    recall.append(recall_score(trueLabels, thPredictions, average='weighted'))

plt.plot(recall, precision, color= 'k', marker= 'o', label= 'Diri')
plt.show()

np.savez('../data/aysPR{}'.format(modelType), np.array(precision), np.array(recall))
print ("Finish saving files!!")

