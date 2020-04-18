import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from Modelj import *

npzfile10 = np.load('./Dataset/test.npz')
feaTh10 = torch.from_numpy(npzfile10['data'] / 255.0).type(torch.FloatTensor)
tarTh10 = torch.from_numpy(npzfile10['target']).type(torch.LongTensor)

def softmax_evidence(logits):
    return F.softmax(logits, dim= 1)
def relu_evidence(logits):
    return F.relu(logits)

numOfClass = 5

modelType = '5CelQuickdraw9'

model = CNNModel(numOfClass)
model.load_state_dict(torch.load('./criticalData/model{}.pt'.format(modelType)))
model.eval()

##################################
outputs10 = model(feaTh10.view(feaTh10.shape[0], 1, 28, 28))
outputs10 = outputs10.detach()
np.savez('./data/test5CelQuickdraw10outputs.npz', outputs10)
np.savez('./data/test5CelQuickdraw10realLabel.npz', npzfile10['target'])
###################################################

evidence = softmax_evidence(outputs10)
predictions = evidence.data.max(1)[1] 
acc = float(predictions.eq(tarTh10.data).sum()) / predictions.shape[0]
print ('The acc of test dataset is {}'.format(100 * acc))

probabilityErr = 1 - torch.max(evidence, dim= 1).values
uevidence = relu_evidence(outputs10)
alpha = uevidence + 1
uncertainty = numOfClass / torch.sum(alpha, dim = 1, keepdims = True)

cntPre = np.zeros(10)
proErrListPre = np.zeros(10)
uncerListPre = np.zeros(10)
pos = 0
for i in predictions.numpy():
    cntPre[i] += 1
    proErrListPre[i] += probabilityErr[pos]
    uncerListPre[i] += uncertainty[pos]
    pos += 1

cntReal = np.zeros(10)
proErrListReal = np.zeros(10)
uncerListReal = np.zeros(10)
pos = 0
for i in tarTh10:  
    cntReal[i] += 1
    proErrListReal[i] += probabilityErr[pos]
    uncerListReal[i] += uncertainty[pos]
    pos += 1

print ('class&numOfInsR&proErrR&uncertaintyR&numOfInsP&proErrP&uncertaintyP\\\\')
for i in range(10):
    print ("%1d  &  %4d  &  %5.2f  &  %5.2f &  %4d  &  %5.2f  &  %5.2f \\\\" \
        %(i, cntReal[i], proErrListReal[i], uncerListReal[i],\
        cntPre[i], proErrListPre[i], uncerListPre[i]))

proErrListReal /= cntReal
uncerListReal /= cntReal

proErrListPre /= cntPre
uncerListPre /= cntPre

for i in range(numOfClass):
    if (cntReal[i] == 0):
        proErrListReal[i] = 1
        uncerListReal[i] = 1
    if (cntPre[i] == 0):
        proErrListPre[i] = 1
        uncerListPre[i] = 1

plt.plot(proErrListReal, marker='*', c='black', label= 'ProbabilityOfErrReal')
plt.plot(uncerListReal, marker='s', c='red', label= 'UncertaintyReal')
plt.plot(proErrListPre, marker='>', c='blue', label= 'ProbabilityOfErrPre')
plt.plot(uncerListPre, marker='o', c='green', label= 'UncertaintyPre')
plt.xticks(np.arange(numOfClass))
plt.xlabel('class')
plt.ylabel('value')
plt.title("Cel5")
plt.legend()
plt.show()