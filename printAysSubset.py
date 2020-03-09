import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# Cel20, Cel20Noise20, Cel20Noise40, Cel20Noise60
# Diri20, Diri20Noise20, Diri20Noise40, Diri20Noise60 
modelType = 'Diri20' 

thsd = np.load('../criticalData/thsd_uncertain.npz')['arr_0']
npzfile = np.load('../data/ays{}.npz'.format(modelType))
#npzfile = np.load('../criticalData/aysRoatedNum.npz')
belief = npzfile['arr_0']
acc = npzfile['arr_1']

fig, axe = plt.subplots()
axe.plot(thsd, belief, color= 'r', marker= '>', label= 'candidate')
axe.set_xlabel('uncertainty threshold')
axe.set_ylabel('number of candidates')
axe.legend(loc= 2)

axe1 = axe.twinx()
axe1.plot(thsd, acc, color= 'k', marker= 'o', label= 'accuracy')
axe1.set_ylabel('accuracy')
axe1.legend(loc= 4)

plt.show()
