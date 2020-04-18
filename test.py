import numpy as np
#import pdb
import torch
import matplotlib.pyplot as plt

#outputs = np.load('./data/test5CelQuickdraw10outputs.npz')
#labels = np.load('./data/test5CelQuickdraw10realLabel.npz')
outputs = np.load('./data/test5DiriQuickdraw10outputs.npz')
labels = np.load('./data/test5DiriQuickdraw10realLabel.npz')
outputs = outputs['arr_0']
labels = labels['arr_0']
outputs = torch.from_numpy(outputs)
labels = torch.from_numpy(labels)
#pdb.set_trace()
outputs = torch.clamp(outputs, min=0)
labels = torch.clamp(labels, min=0)
# Compute softmax
Soutputs = torch.sum(outputs,1)
Soutputs = Soutputs.repeat(5,1)
Soutputs = Soutputs.transpose(1,0)
outputsSM = outputs/Soutputs
# get labels and max prob
Prob = torch.max(outputsSM,1).values
Pred = torch.max(outputsSM,1).indices
Pr_err = 1-Prob
# Now compute uncertaintly 
alpha = outputs+1
#pdb.set_trace()
Sa = torch.sum(alpha,1)
K = outputs.shape[1]
U = K/Sa
for l in range(10):
  plt.subplot(2,5,l+1)
  pr = U.numpy()[np.where(labels==l)]
  plt.hist(pr,bins = 10)
  ti = 'true label ' + str(l)
  plt.axis([0,1,0,1000])
  plt.title(ti)
plt.show()
