import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
import pdb

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
################
#Start main
################
# Loadding testing set
npzfile = np.load('test.npz')
features_test = npzfile['arr_0']
# Add some noise to the feature_test
Amp = 0.2 #between 0 and 1
noise = np.random.randn(features_test.shape[0],features_test.shape[1])
pdb.set_trace()
features_test = features_test + noise.astype(dtype = 'float32')
features_test = np.clip(features_test,0,1)
#plt.plot(SclBe,SaccBe)
targets_test = npzfile['arr_1']
# Create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) 
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)
# Dataloader
batch_size = 100
n_iters = 1
num_epochs = n_iters / (len(features_test) / batch_size)
num_epochs = int(num_epochs)
test_loader = torch.utils.data.DataLoader(test, 
              batch_size = batch_size, 
              shuffle = False)
# Define NN model
model = CNNModel()
# Loading model
model.load_state_dict(torch.load('model.pt'))
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
np.savez('outputs_test.npz', outputsave.numpy(), labelssave.numpy())

