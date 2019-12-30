"""
It loads the model and the testing set 
(because the split between train and test is achieved into the training function) 
and generates a table with beliefs and probabilities. 
In order to provide more challenging images, 
I generate an additional Gaussian noise on pixels gray levels
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
# from torch.utils.tensorboard import SummaryWriter
# import pdb

keepProb = .5
# Create CNN Model
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
        self.dropout = nn.Dropout(keepProb)

    def forward(self, input):
        out1 = self.maxPool(self.relu(self.conv1(input)))
        out2 = self.maxPool(self.relu(self.conv2(out1)))

        #out3 = self.fc1(out2.view(out2.size(0), -1))
        #out3 = self.relu(self.fc1(out2.view(out2.size(0), -1))
        out3 = self.dropout(self.relu(self.fc1(out2.view(out2.size(0), -1))))
        out4 = self.fc2(out3)
        return out4
# Define a class CNNmodelSf with the classical softmax
class CNNModelSf(nn.Module):
    def __init__(self):
        super(CNNModelSf, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 20, stride= 1, 
                                                kernel_size= 5, padding= 0)
        self.conv2 = nn.Conv2d(in_channels= 20, out_channels= 50, stride= 1, 
                                                kernel_size= 5, padding= 0)

        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size= 2)

    def forward(self, input):
        out1 = self.maxPool(self.relu(self.conv1(input)))
        out2 = self.maxPool(self.relu(self.conv2(out1)))

        out3 = self.fc1(out2.view(out2.size(0), -1))
        out = self.fc2(out3)
        return out

################
#Start main
################
# Loadding testing set
npzfile = np.load('../data/test.npz')
features_test = npzfile['arr_0']

# Add some noise to the feature_test
Amp = 0.2 #between 0 and 1
noise = np.random.randn(features_test.shape[0], features_test.shape[1])
# pdb.set_trace()
features_test = features_test + noise.astype(dtype = 'float32')
features_test = np.clip(features_test,0,1)
targets_test = npzfile['arr_1']

# Create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) 
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

batch_size = 100
n_iters = 1
num_epochs = n_iters / (len(features_test) / batch_size)
num_epochs = int(num_epochs)

# Dataloader
test_loader = torch.utils.data.DataLoader(test, 
                                                                   batch_size = batch_size, 
                                                                   shuffle = False)
# Define NN model
model = CNNModel()
#model = CNNModelSf()

# Loading model
model.load_state_dict(torch.load('../data/model.pt'))
#model = torch.load('model.pth')
# Infer on testing set
correct = 0
total = 0
outputsave = []
labelssave = []
for images, labels in test_loader:
    with torch.no_grad():
          test = images.view(100, 1, 28, 28)
          # Forward propagation
          outputs = model(test)
          outputsave.append(outputs)
          labelssave.append(labels)
          # Get predictions from the maximum value
          _, predicted = torch.max(outputs.data, 1)
          # Total number of labels
          total += len(labels)
          correct += (predicted == labels).sum()


outputsave = torch.cat(outputsave, dim=0)
labelssave = torch.cat(labelssave, dim=0)
accuracy = 100 * correct.numpy() / float(total)
print(f'accuracy on testing set: {accuracy}')
# Save output and labels
np.savez('../data/outputs_test.npz', outputsave.numpy(), labelssave.numpy())


