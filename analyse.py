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
# (8400, 10)
outputs = npzfile['arr_0']
# (8400)
targets = npzfile['arr_1']

# Apply Relu
outputs = outputs * (outputs > 0) 

# Comp prob
# (8400)
Se = np.sum(outputs, axis=1)
# (10, 8400), column is same
Se = np.expand_dims(Se, axis=0)
# (8400, 10)
Se = np.repeat(Se, outputs.shape[1], axis = 0)
Se = np.transpose(Se)
# (8400, 10)
Pr = outputs / Se

# Comp Belief
alpha = outputs + 1
Sa = np.sum(alpha,axis=1)
Sa = np.expand_dims(Sa, axis=0)
Sa = np.repeat(Sa, alpha.shape[1],axis = 0)
# (8400, 10)
Sa = np.transpose(Sa)
Be = outputs / Sa

# Comp uncertaintly
K = outputs.shape[1]
#(8400)
U = K / Sa[:,0]

######################################JrX-Add-S#############
# print ('Calculating...')
# numClass = 10
# # (8400, 10)
# Be = outputs
# evidence = Be * (Be > 0)
# alpha = torch.from_numpy(evidence + 1)
# # (8400, 10)
# Pr = (alpha / torch.sum(alpha, dim = 1, keepdims = True)).numpy()
# # (8400, 1)
# U = (numClass / torch.sum(alpha, dim = 1, keepdims = True)).numpy()
######################################JrX-Add-E#############

# Sort belief and prob
classes = []
beliefs = []
probas = []
# Be (8400,10)
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
# conver list to np.array (8400, 10)
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
# (8400, 10)
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
plt.plot(SclPr,SaccPr,label='Using Prob threshold')
plt.legend()
plt.title('Accuracy vs Mean nb of candidates')
plt.show()
#pdb.set_trace()
