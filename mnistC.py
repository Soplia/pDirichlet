"""
The training function
It generates a model and saves as a file
This training is the one with a cross-entropy training error 
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
# import pdb
'''
# Clean the tesnsorboardX's root dir
import os
import shutil
if os.path.isdir('./mnistC'):
    shutil.rmtree('./mnistC')
    print('Finish deleting')
'''

train = pd.read_csv("../data/train.csv", dtype = np.float32)
# Split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = train.label.values
# Normalization
features_numpy = train.loc[:, train.columns != "label"].values / 255

# Train test split.  Size of train data is 80% and size of test data is 20%.
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                                                                    targets_numpy,
                                                                                                                    test_size = 0.1,
                                                                                                                    random_state = 42) 
# pdb.set_trace()
# Create feature and targets tensor for train set.
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) 

# Create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) 

# batch_size, epoch and iteration
batch_size = 100
n_iters = 50000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)
print(f'{num_epochs} epochs in training')

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

# Data loader
train_loader = torch.utils.data.DataLoader(train, 
                                                                     batch_size = batch_size, 
                                                                     shuffle = False)
test_loader = torch.utils.data.DataLoader(test, 
                                                                   batch_size = batch_size, 
                                                                   shuffle = False)

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        
        return out
    
# Create ANN
model = CNNModel()
# Cross Entropy Loss
error = nn.CrossEntropyLoss()
# SGD Optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
writer = SummaryWriter('../boardx/mnistC')

for epoch in range(num_epochs):
    print('Training-epoch: {}'.format(epoch + 1))
    for i, (images, labels) in enumerate(train_loader):
        train = images.view(100,1,28,28)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)       
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()

# save model
torch.save(model.state_dict(),'../data/model.pt')
# save testing dataset
np.savez('../data/test.npz', features_test, targets_test)
#torch.save(model,'model.pth')

#count += 1
#if count % 50 == 0:
#    # Calculate Accuracy
#    correct = 0
#    total = 0
#    # Iterate through test dataset
#    for images, labels in test_loader:
#      test = images.view(100, 1, 28, 28)
#      # Forward propagation
#      outputs = model(test)
#      # Get predictions from the maximum value
#      _, predicted = torch.max(outputs.data, 1)
#      # Total number of labels
#      total += len(labels)
#      correct += (predicted == labels).sum()
#    accuracy = 100 * correct / float(total)
#    writer.add_scalar(tag = 'Accuracy-CNN',
#      scalar_value= accuracy.data, global_step= epoch * len(train_loader) + i)
#    writer.add_scalar(tag = 'Loss-CNN', 
#      scalar_value= loss.data, global_step= epoch * len(train_loader) + i)
#    # store loss and iteration
#    loss_list.append(loss.data)
#    iteration_list.append(count)
#    accuracy_list.append(accuracy)
#    if count % 500 == 0:
#       # Print Loss
#       print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.item(), accuracy))
## visualization loss

#plt.plot(iteration_list,loss_list)
#plt.xlabel("Number of iteration")
#plt.ylabel("Loss")
#plt.title("CNN: Loss vs Number of iteration")
#plt.show()

## visualization accuracy
#plt.plot(iteration_list,accuracy_list,color = "red")
#plt.xlabel("Number of iteration")
#plt.ylabel("Accuracy")
#plt.title("CNN: Accuracy vs Number of iteration")
#plt.show()

########################## JrX-Add-S #######################
## features: 100(Ins) * 10(Labels)
#def DirichletFunction(inputs):
#    features = inputs.detach()
#    numLabels = 10
    
#    #|prediction|belief|dirichlet|
#    preBefDirMatrix = np.zeros((numLabels, 3), 
#                                dtype = 'float')

#    # Start loop
#    for prediction in features:
#        prediction = np.array(prediction)
        
#        K = len(prediction)

#        alpha = prediction + 1
#        S = sum(alpha)

#        belief = prediction / S
#        dirichlet = alpha / S
#        uncertain = K / S 
        
#        '''
#        print('=========================')
#        print(f'{K} classes')
#        print('Prediction = ', prediction)
#        print('Belief = ', belief)
#        print('DirlietchD = ', dirichlet)
#        print('Uncertaint = ', uncertain)
#        '''

#        ibs = np.flip(np.argsort(dirichlet))
#        ds = dirichlet[ibs]
#        ps = prediction[ibs]
#        bs = belief[ibs]

#        cSumPs = np.cumsum(ps)
#        cSumDs = np.cumsum(ds)
#        cSumBs = np.cumsum(bs)

#        preBefDirMatrix[:, 0] = preBefDirMatrix[:, 0] + cSumPs
#        preBefDirMatrix[:, 1] = preBefDirMatrix[:, 1] + cSumBs
#        preBefDirMatrix[:, 2] = preBefDirMatrix[:, 2] + cSumDs

#    return preBefDirMatrix
########################## JrX-Add-E #######################
#correct = 0
#total = 0
#numOfLabels = 10
#preBefDirMatrix = np.zeros((numOfLabels, 3), 
#                                                dtype = 'float')
 
## Iterate through test dataset
#for images, labels in test_loader:
#    test = images.view(100, 1, 28, 28)
## Forward propagation
#    outputs = model(test)

########################## JrX-Add-S #######################
#    tmp = DirichletFunction(outputs)
#    preBefDirMatrix = preBefDirMatrix + tmp
########################## JrX-Add-E #######################

#    # Get predictions from the maximum value
#    _, predicted = torch.max(outputs.data, 1)
#    # Total number of labels
#    total += len(labels)
#    correct += (predicted == labels).sum()
           
#accuracy = 100 * correct / float(total)
#accuracy_list.append(accuracy)
#print('Accuracy: {} %'.format(accuracy))

########################## JrX-Add-S #######################
#preBefDirMatrix = preBefDirMatrix / total
#print(preBefDirMatrix)

#tmp = np.ones(numOfLabels, dtype= int)
#idx = np.cumsum(tmp)
#for i in range(1, 3):
#    plt.plot(idx, preBefDirMatrix[:, i].T)
#plt.xlabel("Number of Candidates")
#plt.ylabel("Probability")
#plt.title("Belief-Dirichlet")
#plt.legend(['Belief', 'Dirichlet'])
#plt.show()
########################## JrX-Add-E #######################

