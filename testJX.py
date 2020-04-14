import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import torch
import matplotlib.pyplot as plt

def softmax_evidence(logits):
    return F.softmax(logits, dim= 1)
def relu_evidence(logits):
    return F.relu(logits)

outputs = np.load('../data/testUPED5outputs.npz')
labels = np.load('../data/testUPED5realLabel.npz')
outputs = outputs['arr_0']
labels = labels['arr_0']
outputs = torch.from_numpy(outputs)
labels = torch.from_numpy(labels)

# Clamp all elements in input into the range [ min, max ] 
outputs = torch.clamp(outputs, min=0)
labels = torch.clamp(labels, min=0)
# Compute softmax
Soutputs = torch.sum(outputs, 1)
Soutputs = Soutputs.repeat(5,1)
Soutputs = Soutputs.transpose(1,0)
outputsSM = outputs/Soutputs

# get labels and max prob
prob = torch.max(outputsSM, 1).values
pred = torch.max(outputsSM, 1).indices
pr_err = 1-prob
plt.plot(pred.numpy()+1 ,pr_err.numpy() ,'+r')
plt.title('error prob vs label')
plt.show()

outputsJR = softmax_evidence(relu_evidence(outputs))
Prob = torch.max(outputsJR, 1).values
Pred = torch.max(outputsJR, 1).indices
Pr_err = 1-Prob
plt.plot(Pred.numpy()+1 ,Pr_err.numpy() ,'+r')
plt.title('error prob vs label')
plt.show()
