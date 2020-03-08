import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
targets_numpy = np.argmax(mnist.train.labels, axis= 1)
features_numpy = mnist.train.images 

features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                                                                    targets_numpy,
                                                                                                                    test_size = 0.2,
                                                                                                                    random_state = 42) 
# Create feature and targets tensor for train set.
featuresTrain = torch.from_numpy(features_numpy)
targetsTrain = torch.from_numpy(targets_numpy).type(torch.LongTensor) 

# Utility parameters
epochs = 20
batch_size = 100
numOfClass = 10
learning_rate = 0.1
print(f'{epochs} epochs in training')

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
# Data loader
train_loader = torch.utils.data.DataLoader(train, 
                                                                     batch_size = batch_size, 
                                                                     shuffle = False)

def softmax_evidence(logits):
    return F.softmax(logits)

# Define a class CNNmodelSf with the classical softmax
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
        #all rows add up to 1
        self.softmax = nn.Softmax(dim= 1)

    def forward(self, input):
        out1 = self.maxPool(self.relu(self.conv1(input)))
        out2 = self.maxPool(self.relu(self.conv2(out1)))
        out3 = self.dropout(self.relu(self.fc1(out2.view(out2.size(0), -1))))
        out4 = self.softmax(self.fc2(out3))
        return out4

model = CNNModel()
# SGD Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
error = nn.CrossEntropyLoss();

# CNNModel with softmax cross entropy loss function
acc1d = []
loss1d = []
fig, axes = plt.subplots(nrows= 2, ncols= 1)
for epoch in range(epochs):
    print('Training-epoch: {}'.format(epoch + 1))
    for i, (images, labels) in enumerate(train_loader):
        train = images.view(batch_size, 1, 28, 28)
        optimizer.zero_grad()
        outputs = model(train) 

        acc = torch.sum(torch.argmax(outputs, dim= 1) ==
                labels).item() / float(labels.shape[0])
        acc1d.append(acc)

        loss = error(outputs, labels)
        loss1d.append(loss)
        loss.backward()
        optimizer.step()

print ('Finish Training')

# save model
torch.save(model.state_dict(), '../criticalData/modelCel{}.pt'.format(epochs))

print ('Finish Saving Files')
axes[0].plot(acc1d, label= 'Accuracy')
axes[1].plot(loss1d, label= 'Loss')
axes[0].set_ylabel('AccVal')
axes[1].set_ylabel('LossVal')
axes[1].set_xlabel('Iteration')
axes[0].legend()
axes[1].legend() 
plt.show()