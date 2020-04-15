import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

npzfile10 = np.load('./Dataset/test.npz')
feaTh10 = torch.from_numpy(npzfile10['data'] / 255.0).type(torch.FloatTensor)
tarTh10 = torch.from_numpy(npzfile10['target']).type(torch.LongTensor)

def softmax_evidence(logits):
    return F.softmax(logits, dim= 1)

numOfClass = 5

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
        self.fc2 = nn.Linear(500, numOfClass)

        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size= 2)
        self.dropout = nn.Dropout(.5)

    def forward(self, input):
        out1 = self.maxPool(self.relu(self.conv1(input)))
        out2 = self.maxPool(self.relu(self.conv2(out1)))
        out3 = self.dropout(self.relu(self.fc1(out2.view(out2.size(0), -1))))
        out4 = self.fc2(out3)
        return out4

modelType = '5DiriQuickdraw9'

model = CNNModel()
model.load_state_dict(torch.load('./criticalData/model{}.pt'.format(modelType)))
model.eval()

##################################
outputs10 = model(feaTh10.view(feaTh10.shape[0], 1, 28, 28))
outputs10 = outputs10.detach()
np.savez('./data/test5DiriQuickdraw10outputs.npz', outputs10)
np.savez('./data/test5DiriQuickdraw10realLabel.npz', npzfile10['target'])
###################################################

evidence = softmax_evidence(outputs10)
predictions = evidence.data.max(1)[1]
acc = float(predictions.eq(tarTh10.data).sum()) / predictions.shape[0]
print ('The acc of test dataset is {}'.format(100 * acc))

print (torch.sum(evidence, dim= 1))
print (torch.sum(evidence, dim= 1).shape)
print (torch.max(evidence, dim= 1))
#Calculate uncertainty for class

probabilityErr = 1 - torch.max(evidence, dim= 1).values
uevidence = relu_evidence(outputs10)
alpha = uevidence + 1
uncertainty = numOfClass / torch.sum(alpha, dim = 1, keepdims = True)

cnt = np.zeros(10)
probabilityErrList = np.zeros(10)
uncertaintyList = np.zeros(10)

# 根据真实标签进行统计
# 各个类别（10）的实例数目
# according to real label count the instance number 
# of each class
for i in tarTh10:
    cnt[i] = cnt[i ] + 1

# 根据实际预测
# 统计各个类别（10）的错误预测率
# according to prediction count probabilityErr
# and uncertainty
pos = 0
for i in predictions.numpy():
    probabilityErrList[i] = probabilityErrList[i] + probabilityErr[pos]
    pos = pos + 1

pos = 0
for i in predictions.numpy():
    uncertaintyList[i] = uncertaintyList[i] + uncertainty[pos]
    pos = pos + 1
#for i in np.arange(numOfClass):
#    print (np.sum(tarTh.numpy() == i), end =" , ")
#    print (np.sum(predictions.numpy() == i))

for i in range(10):
    print ("{}  &  {}  &  {}  &  {} \\\\".format(i, cnt[i], probabilityErrList[i], uncertaintyList[i]))

probabilityErrList = probabilityErrList / cnt
uncertaintyList = uncertaintyList / cnt

for i in range(numOfClass):
    if (cnt[i] == 0):
        probabilityErrList[i] = 1
        uncertaintyList[i] = 1

plt.plot(probabilityErrList, marker='*', c='black', label= 'ProbabilityOfErr')
plt.plot(uncertaintyList, marker='s', c='red', label= 'Uncertainty')
plt.xticks(np.arange(numOfClass))
plt.xlabel('class')
plt.ylabel('value')
plt.title("Diri5")
plt.legend()
plt.show()
#np.savez('../data/test{}.npz'.format(modelType), outputs.numpy(), tarTh.numpy())


