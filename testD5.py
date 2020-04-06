import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import rotateImg 

npzfile = np.load('../data/testRaw.npz')

feaTh = torch.from_numpy(npzfile['arr_0']).type(torch.float)
tarTh = torch.from_numpy(npzfile['arr_1']).type(torch.int)

def softmax_evidence(logits):
    return F.softmax(logits, dim= 1)

def relu_evidence(logits):
    return F.relu(logits)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 20, stride= 1, 
                                                kernel_size= 5, padding= 0)
        self.conv2 = nn.Conv2d(in_channels= 20, out_channels= 50, stride= 1, 
                                                kernel_size= 5, padding= 0)

        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size= 2)
        self.dropout = nn.Dropout(.5)

    def forward(self, input):
        out1 = self.maxPool(self.relu(self.conv1(input)))
        out2 = self.maxPool(self.relu(self.conv2(out1)))
        out3 = self.dropout(self.relu(self.fc1(out2.view(out2.size(0), -1))))
        out4 = self.fc2(out3)
        return out4

modelType = '5Cel9' # Diri20, Cel20
numClass = 10

model = CNNModel()
model.load_state_dict(torch.load('../criticalData/model{}.pt'.format(modelType)))
model.eval()

outputs = model(feaTh.view(feaTh.shape[0], 1, 28, 28))
outputs = outputs.detach()
predictions = torch.argmax(outputs.data, dim= 1)
acc = (predictions == tarTh).sum() / float(predictions.shape[0])
print ('The acc of test dataset is {}'.format(100 * acc))

for i in np.arange(10):
    print (np.sum(tarTh.numpy() == i), end =" , ")
    print (np.sum(predictions.numpy() == i))
#Calculate uncertainty for class

evidence = relu_evidence(outputs)
alpha = evidence + 1
u = numClass / torch.sum(alpha, dim = 1, keepdims = True)

cnt = np.zeros(10)
lu = np.zeros(10)

for i in predictions.numpy():
    cnt[i] = cnt[i] + 1

pos = 0
for i in predictions.numpy():
    lu[i] = lu[i] + u[pos]
    pos = pos + 1

lu = lu / cnt

for i in range(10):
    if (cnt[i] == 0):
        lu[i] = 1


plt.plot(lu, marker='*', c='black', label= 'uncertainty')
plt.xticks(np.arange(10))
plt.xlabel('class')
plt.ylabel('uncertainty')
plt.title('Diri5-Uncertainty')
plt.show()
#np.savez('../data/test{}.npz'.format(modelType), outputs.numpy(), tarTh.numpy())
