import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#modelType = 'Cel20' 
#npzfile = np.load('../data/aysPR{}.npz'.format(modelType))
##npzfile = np.load('../data/aysSubsetPR{}.npz'.format(modelType))
#precisionc = npzfile['arr_0']
#recallc = npzfile['arr_1']

modelType = 'Diri20' 
npzfile = np.load('../data/aysPR{}.npz'.format(modelType))
#npzfile = np.load('../data/aysSubsetPR{}.npz'.format(modelType))
precisiond = npzfile['arr_0']
recalld = npzfile['arr_1']

#plt.plot(precisiond, color= 'k', marker= 'o', label= 'precision')
#plt.plot(recalld, color= 'r', marker= '*', label= 'recall')

#plt.plot(recallc, precisionc, color= 'r', marker= '>', label= 'Cel')
plt.plot(recalld, precisiond, color= 'k', marker= 'o', label= 'Diri')
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()

plt.show()

