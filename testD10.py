import numpy as np
#import pdb
import torch
import matplotlib.pyplot as plt

def RocPrErr(fnout, fnlab):
  # Pr_Err: Compute roc curve from output file and true labels
  outputs = np.load(fnout)
  labels = np.load(fnlab)
  outputs = outputs['arr_0']
  labels = labels['arr_0']
  outputs = torch.from_numpy(outputs).type(torch.FloatTensor)
  labels = torch.from_numpy(labels).type(torch.LongTensor)
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
  # Now compute ROC curve with rejection strategy
  estlabels = torch.argmax(outputs,1)
  # Compute True detection rate
  vtd =(estlabels == labels) 
  npos = torch.sum(vtd)
  pos_rate = (100.0 * npos)/len(labels)
  print("accuracy = {}".format(pos_rate))
  # Compute ROC Curve
  # sort according to U and Pr_err
  sp = torch.argsort(Pr_err)
  npp = vtd[sp]
  CSnpp = torch.cumsum(npp,0).type(torch.DoubleTensor)
  CS = torch.cumsum(torch.ones(len(sp)),0).type(torch.DoubleTensor)
  nbtotposPr = torch.sum(labels<5,0).type(torch.DoubleTensor)
  True_Pos_Rate_Pr_Vec = CSnpp/nbtotposPr
  False_Pos_Rate_Pr_Vec = 1- CSnpp/CS
  return True_Pos_Rate_Pr_Vec, False_Pos_Rate_Pr_Vec
  

def RocU(fnout, fnlab):
  # Uncertainty: Compute roc curve from output file and true labels
  outputs = np.load(fnout)
  labels = np.load(fnlab)
  outputs = outputs['arr_0']
  labels = labels['arr_0']
  outputs = torch.from_numpy(outputs).type(torch.FloatTensor)
  labels = torch.from_numpy(labels).type(torch.LongTensor)
  #pdb.set_trace()
  outputs = torch.clamp(outputs, min=0)
  labels = torch.clamp(labels, min=0)
  # Now compute uncertaintly 
  alpha = outputs+1
  Sa = torch.sum(alpha,1)
  K = outputs.shape[1]
  U = K/Sa
  # Now compute ROC curve with rejection strategy
  estlabels = torch.argmax(outputs,1)
  # Compute True detection rate
  vtd =(estlabels == labels) 
  npos = torch.sum(vtd)
  pos_rate = (100.0 * npos)/len(labels)
  print("accuracy = {}".format(pos_rate))
  # Compute ROC Curve
  # sort according to U 
  sU = torch.argsort(U)
  npU = vtd[sU]
  CSnpU = torch.cumsum(npU,0).type(torch.DoubleTensor)
  CS = torch.cumsum(torch.ones(len(sU)),0).type(torch.DoubleTensor)
  nbtotposU = torch.sum(labels<5,0).type(torch.DoubleTensor)
  True_Pos_Rate_U_Vec = CSnpU/nbtotposU
  False_Pos_Rate_U_Vec = 1- CSnpU/CS
  return True_Pos_Rate_U_Vec, False_Pos_Rate_U_Vec

###
###
# Compute Pr_Err ROC curve for Cross Entropy
tp_P_CE, fp_P_CE = RocPrErr('./data/test5CelQuickdraw10outputs.npz','./data/test5CelQuickdraw10realLabel.npz' )
# Compute Pr_Err ROC curve for Dirichlet 
tp_P_DI, fp_P_DI = RocPrErr('./data/test5DiriQuickdraw10outputs.npz','./data/test5DiriQuickdraw10realLabel.npz')
# Compute Uncert ROC curve for Cross Entropy
tp_U_CE, fp_U_CE = RocU('./data/test5CelQuickdraw10outputs.npz','./data/test5CelQuickdraw10realLabel.npz' )
# Compute Uncert ROC curve for Dirichlet 
tp_U_DI, fp_U_DI = RocU('./data/test5DiriQuickdraw10outputs.npz','./data/test5DiriQuickdraw10realLabel.npz')

#pdb.set_trace()

# Display
plt.subplot(131)
plt.plot(fp_P_CE,tp_P_CE,'-r',label='Cross Entropy')
plt.plot(fp_P_DI,tp_P_DI,'-g',label='Dirichlet')
plt.title('Error Probabiliy ROC curves')
plt.axis([0,1,0,1])
plt.xlabel('False Pos.')
plt.ylabel('True Pos.')
plt.legend()
plt.subplot(132)
plt.plot(fp_U_CE,tp_U_CE,'-r',label='Cross Entropy')
plt.plot(fp_U_DI,tp_U_DI,'-g',label='Dirichlet')
plt.title('Uncertaintly ROC curves')
plt.axis([0,1,0,1])
plt.xlabel('False Pos.')
plt.ylabel('True Pos.')
plt.legend()
plt.subplot(133)
plt.plot(fp_P_CE,tp_P_CE,'-r',label='Pr Err Cross Entropy')
plt.plot(fp_U_DI,tp_U_DI,'-g',label='Unc Dirichlet')
plt.title('Unc/Dir and Pr_err/CE ROC curves')
plt.axis([0,1,0,1])
plt.xlabel('False Pos.')
plt.ylabel('True Pos.')
plt.legend()
plt.show()
'''
for l in range(10):
  plt.subplot(2,5,l+1)
  #pr = Pr_err.numpy()[np.where(labels==l)]
  pr = U.numpy()[np.where(labels==l)]
  plt.hist(pr,bins = 10)
  ti = 'Uncer hist for true label ' + str(l)
  plt.axis([0,1,0,1000])
  plt.title(ti)
plt.show()
'''
