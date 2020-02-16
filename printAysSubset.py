import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

npzfile = np.load('../data/aysDiri.npz')
belief = npzfile['arr_0']
acc = npzfile['arr_1']
thsd = np.load('../data/thsd_uncertain.npz')['arr_0']

fig, axe = plt.subplots()
axe.plot(thsd, belief, color= 'r', marker= '>', label= 'candinators')
axe.set_xlabel('uncertain_thsd')
axe.set_ylabel('num of candinators')
axe.legend(loc= 2)

axe1 = axe.twinx()
axe1.plot(thsd, acc, color= 'k', marker= 'o', label= 'acc')
axe1.set_ylabel('accuracy')
axe1.legend(loc= 4)

plt.show()
