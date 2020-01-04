"""
it analyses and displays a curve given the accuracy according to the mean number 
if samples needed (this curve has been generated using a threshold on belief/proba)
"""
import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
# import pdb

# Read Data
npzfile = np.load('../data/outputs_test.npz')
outputs = npzfile['arr_0']
targets = npzfile['arr_1']

# Apply Relu
outputs = outputs * (outputs > 0) 

# Comp prob
Se = np.sum(outputs,axis=1)
Se = np.expand_dims(Se, axis=0)
Se = np.repeat(Se, outputs.shape[1],axis = 0)
Se = np.transpose(Se)
Pr = outputs / Se


# evidence = relu_evidence(outputs)
# alpha = evidence + 1
# u = numOfClass / torch.sum(alpha, dim = 1, keepdims = True)
# pro = alpha / torch.sum (alpha, dim = 1, keepdims = True)

# Comp Belief
alpha = outputs + 1
Sa = np.sum(alpha,axis=1)
Sa = np.expand_dims(Sa, axis=0)
Sa = np.repeat(Sa, alpha.shape[1],axis = 0)
Sa = np.transpose(Sa)
Be = outputs / Sa

# Comp uncertaintly
K = outputs.shape[1]
U = K / Sa[:,0]

# Sort belief and prob
classes = []
beliefs = []
probas = []
for i in range(Be.shape[0]):
      oneBe = Be[i,:]
      onePr = Pr[i,:]
      ibs = np.flip(np.argsort(Be[i,:]))
      Bes = oneBe[ibs]
      Prs = onePr[ibs]
      classes.append(ibs)
      beliefs.append(Bes)
      probas.append(Prs)

classes = np.concatenate(classes, axis=0)
classes = classes.reshape(outputs.shape)
beliefs = np.concatenate(beliefs, axis=0)
beliefs = beliefs.reshape(outputs.shape)
probas = np.concatenate(probas, axis=0)
probas = probas.reshape(outputs.shape)

#belief and proba CumSum
beliefsCS = np.cumsum(beliefs,axis=1)
probasCS = np.cumsum(probas,axis=1)

# Loop on Be/Pr theshold
####
## +1 to all classes because 0 should not be classe
####
targetsTable = (targets + 1)
targetsTable = np.expand_dims(targetsTable , axis=0)
targetsTable = np.repeat(targetsTable, outputs.shape[1],axis = 0)
targetsTable = np.transpose(targetsTable)
classes = classes + 1
nb = 100 # Nb of points
SclBe = []
SclPr = []
SaccBe = []
SaccPr = []
for th in  np.linspace(0, 1, num=nb):
      maskBe = beliefsCS <= th
      maskPr = probasCS <= th
      classesBe = maskBe * (classes)
      classesPr = maskPr * (classes)
      nbClBe = np.sum(maskBe,axis=1)
      nbClPr = np.sum(maskPr,axis=1)
      meanClBe = np.mean(nbClBe)
      meanClPr = np.mean(nbClPr)

      # look for classes
      compBe = (classesBe == targetsTable)
      compPr = (classesPr == targetsTable)

      # Get Accuracy
      accBe = sum(compBe.flatten()) / outputs.shape[0] * 100
      accPr = sum(compPr.flatten()) / outputs.shape[0] * 100
      SclBe.append(meanClBe)
      SclPr.append(meanClPr)
      SaccBe.append(accBe)
      SaccPr.append(accPr)
      # print(f'accBe = {accBe}, mean classes = {meanClBe}')
      # print(f'accPr = {accPr}, mean classes = {meanClPr}')

plt.plot(SclBe,SaccBe,label='Using Belief threshold')
plt.plot(SclPr,SaccPr,label='Using SoftMax threshold')
plt.legend()
plt.title('Accuracy vs Mean nb of candidates')
plt.show()
#pdb.set_trace()

