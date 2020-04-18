import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Modelj import *

npzfile = np.load('./Dataset/train.npz')
features_numpy = npzfile['data'] / 255.0
targets_numpy = npzfile['target'] 

numOfClass = 5
epochs = 9
batch_size = 256
learning_rate = 0.1
lmb = 0.005

targets_train = targets_numpy[targets_numpy < numOfClass]
features_train = features_numpy[targets_numpy < numOfClass]

# Create feature and targets tensor for train set.
featuresTrain = torch.from_numpy(features_train).type(torch.FloatTensor)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) 

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
# Data loader
train_loader = torch.utils.data.DataLoader(train, 
                                                                     batch_size = batch_size, 
                                                                     shuffle = True)

print(f'{epochs} epochs in training')

model = CNNModel(numOfClass)

optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
error = nn.CrossEntropyLoss();

acc1d = []
loss1d = []
for epoch in range(epochs):
    print('Training-epoch: {}'.format(epoch + 1))
    for i, (images, labels) in enumerate(train_loader):
        train = images.view(-1, 1, 28, 28)
        optimizer.zero_grad()
        outputs = F.softmax(model(train) , dim= 1)
        
        acc = torch.sum(torch.argmax(outputs, dim= 1) ==
                labels).item() / labels.shape[0]
        acc1d.append(acc)

        loss = error(outputs, labels)

        loss1d.append(loss)
        loss.backward()
        optimizer.step()

print ('Finish Training')

# save model
torch.save(model.state_dict(), './criticalData/model5CelQuickdraw{}.pt'.format(epochs))

print ('Finish Saving Files')

fig, axes = plt.subplots(nrows= 2, ncols= 1)
axes[0].plot(acc1d, label= 'Accuracy')
axes[1].plot(loss1d, label= 'Loss')
axes[0].set_ylabel('AccVal')
axes[1].set_ylabel('LossVal')
axes[1].set_xlabel('Iteration')
axes[0].legend()
axes[1].legend() 
plt.show()