"""
It generates and trains a CNN model with loss function eq5
Saving the state_dict as a file 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

train = pd.read_csv("../data/train.csv", dtype = np.float32)

# Split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = train.label.values
# Normalization
features_numpy = train.loc[:, train.columns != "label"].values / 255
# Train test split.  Size of train data is 90% and size of test data is 10%.
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                                                                    targets_numpy,
                                                                                                                    test_size = 0.2,
                                                                                                                    random_state = 42) 
# Create feature and targets tensor for train set.
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) 

# Utility parameters
epochs = 50
batch_size = 100
numOfClass = 10
# I am not sure about global_step, I adopt 100
global_step = 100
annealing_step = 10 * (len(features_train) // batch_size)
lmb = 0.005
learning_rate = 0.1
print(f'{epochs} epochs in training')

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
# Data loader
train_loader = torch.utils.data.DataLoader(train, 
                                                                     batch_size = batch_size, 
                                                                     shuffle = False)

# KL Divergence calculator
# Shape of Input:    alpha: r × numOfClass; numOfClass: 1 × 1
# Shape of Output: r × 1
def KL(alpha, numOfClass):
    # beta = tf.constant(np.ones((1, K)), dtype= tf.float32)
    beta = torch.ones((1, numOfClass), dtype = torch.float32, requires_grad= False)
    # S_alpha = tf.reduce_sum(alpha, axis=1, keep_dims= True)
    S_alpha = torch.sum(alpha, dim= 1, keepdims= True)
    #S_beta = tf.reduce_sum(beta, axis= 1, keep_dims= True)
    S_beta = torch.sum(beta, dim = 1, keepdims= True)
    # lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha), axis=1, keep_dims=True)
    lnB = torch.lgamma(input= S_alpha) - torch.sum(torch.lgamma(alpha), dim= 1, keepdims= True)
    # lnB_uni = tf.reduce_sum(tf.lgamma(beta), axis= 1,keep_dims= True) - tf.lgamma(S_beta)
    lnB_uni = torch.sum(torch.lgamma(beta), dim= 1, keepdims= True) - torch.lgamma(S_beta)
    
    # dg0 = tf.digamma(S_alpha)
    dg0 = torch.digamma(input= S_alpha)
    #dg1 = tf.digamma(alpha)
    dg1 = torch.digamma(input= alpha)
    
    # kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keep_dims=True) + lnB + lnB_uni
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim = 1, keepdims= True) + lnB + lnB_uni
    return kl

#Shape of Input:    alpha: r × numOfClass; numOfClass: 1 × 1
#                             global_step: 1 × 1; annealing_step: 1 × 1
#                             p: r × numOfClass
# Shape of Output: r × 1
def loss_eq5(p, alpha, numOfClass, global_step, annealing_step):
    # S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    S = torch.sum(alpha, dim = 1, keepdims = True)
    # loglikelihood = tf.reduce_sum((p-(alpha/S))**2, axis=1, keepdims=True) + \
    #                        tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)
    logLikeHood = torch.sum ((p - (alpha / S)) ** 2, dim = 1, keepdims= True) + \
                              torch.sum (alpha * (S - alpha) / (S * S * (S + 1)), dim = 1, keepdims= True)
    # KL_reg =  tf.minimum(1.0, tf.cast(global_step/annealing_step, tf.float32)) * \
    #                  KL((alpha - 1)*(1-p) + 1 , K)
    KL_reg = min(1.0, float(global_step) / annealing_step) * \
                     KL((alpha - 1) * (1 - p) + 1, numOfClass)
    return logLikeHood + KL_reg

#Shape of Input:    alpha: r × numOfClass; numOfClass: 1 × 1
#                             global_step: 1 × 1; annealing_step: 1 × 1
#                             p: r × numOfClass
# Shape of Output: r × 1
def loss_eq4(p, alpha, numOfClass, global_step, annealing_step):
    #loglikelihood = tf.reduce_mean(tf.reduce_sum(p * \
    #                          (tf.digamma(tf.reduce_sum(alpha, axis=1, keepdims=True)) - \
    #                          tf.digamma(alpha)), 1, keepdims=True))
    logLikeHood = torch.mean(torch.sum(p * (torch.digamma(torch.sum(alpha, dim= 1, keepdims= True)) - \
                                                torch.digamma(alpha)), dim= 1, keepdims= True))

    #KL_reg =  tf.minimum(1.0, tf.cast(global_step/annealing_step, tf.float32)) * \
    #                                    KL((alpha - 1)*(1-p) + 1 , K)
    KL_reg = min(1.0, float(global_step) / annealing_step) * \
                     KL((alpha - 1) * (1 - p) + 1 , numOfClass)

    return logLikeHood + KL_reg

#Shape of Input:    alpha: r × numOfClass; numOfClass: 1 × 1
#                             global_step: 1 × 1; annealing_step: 1 × 1
#                             p: r × numOfClass
# Shape of Output: r × 1
def loss_eq3(p, alpha, numOfClass, global_step, annealing_step):
    #loglikelihood = tf.reduce_mean(tf.reduce_sum(p * (tf.log(tf.reduce_sum(alpha, axis=1, keepdims=True)) - tf.log(alpha)), 1, keepdims=True))
    logLikeHood = torch.mean(torch.sum(p * torch.log(torch.sum(alpha, dim= 1, keepdims= True)) - \
                              torch.log(alpha), dim= 1, keepdims= True))

    #KL_reg =  tf.minimum(1.0, tf.cast(global_step/annealing_step, tf.float32)) * KL((alpha - 1)*(1-p) + 1 , K)
    KL_reg = min(1.0, float(global_step) / annealing_step) * \
                     KL((alpha - 1) * (1 - p) + 1, numOfClass)
    return logLikeHood + KL_reg

# Three types evidence
def relu_evidence(logits):
    return F.relu(logits)

def exp_evidence(logits): 
    return torch.exp(logits / 1000)

def softmax_evidence(logits):
    return F.softmax(logits)

#Computes half the L2 norm of a tensor without the sqrt
def L2Loss(inputs):
    return torch.sum(inputs ** 2) / 2

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

    def forward(self, input):
        out1 = self.maxPool(self.relu(self.conv1(input)))
        out2 = self.maxPool(self.relu(self.conv2(out1)))
        out3 = self.dropout(self.relu(self.fc1(out2.view(out2.size(0), -1))))
        out4 = self.fc2(out3)
        return out4

# Create ANN
model = CNNModel()
# SGD Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

# CNNModel with softmax cross entropy loss function
acc1d = []
loss1d = []
fig, axes = plt.subplots(nrows= 2, ncols= 1)
for epoch in range(epochs):
    print('Training-epoch: {}'.format(epoch + 1))
    for i, (images, labels) in enumerate(train_loader):
        # Change the shape of labels from (images.shape[0]) to 
        # (images.shape[0], 10)
        newLabels = torch.zeros((labels.shape[0], 10), dtype= torch.float32)
        cnt = 0
        for pos in labels:
            newLabels[cnt, pos] = 1.0
            cnt += 1

        train = images.view(100, 1, 28, 28)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train) 
        evidence = relu_evidence(outputs)
        alpha = evidence + 1
        # u = numOfClass / torch.sum(alpha, dim = 1, keepdims = True)
        # pro = alpha / torch.sum (alpha, dim = 1, keepdims = True)
        acc = torch.sum(torch.argmax(outputs, dim= 1).view(-1, 1) ==
                torch.argmax(newLabels, dim= 1).view(-1, 1)).item() / newLabels.shape[0]
        acc1d.append(acc)
        
        # Should not input pro, must be the real label
        loss = torch.mean(loss_eq5(newLabels, alpha, numOfClass, global_step, annealing_step))
        l2Loss = (L2Loss(model.state_dict()['fc1.weight']) + 
                         L2Loss(model.state_dict()['fc2.weight'])) * lmb 
        loss1d.append(loss + l2Loss)
        #loss1d.append(loss)

        (loss + l2Loss).backward()
        #(loss).backward()
        optimizer.step()

print ('Finish Training')

# save model
torch.save(model.state_dict(), '../criticalData/modelDiri.pt')
# save accuracy and loss during training the model
torch.save(acc1d, '../criticalData/accTrainDiri.pt')
torch.save(loss1d, '../criticalData/lossTrainDiri.pt')
# save testing dataset
np.savez('../data/testRaw.npz', features_test, targets_test)

print ('Finish Saving Files')

axes[0].plot(acc1d, label= 'Accuracy')
axes[1].plot(loss1d, label= 'Loss')
axes[0].set_ylabel('AccVal')
axes[1].set_ylabel('LossVal')
axes[1].set_xlabel('Iteration')
axes[0].legend()
axes[1].legend() 
plt.show()