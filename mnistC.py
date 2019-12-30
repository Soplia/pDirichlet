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
# import pdb

train = pd.read_csv("../data/train.csv", dtype = np.float32)

# Split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = train.label.values
# Normalization
features_numpy = train.loc[:, train.columns != "label"].values / 255

# Train test split.  Size of train data is 90% and size of test data is 10%.
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

# Utility parameters
n_iters = 5000
batch_size = 100
num_epochs = int(n_iters / (len(features_train) / batch_size))
numOfClass = 10
global_step = 0
n_batches = len(features_train)  // batch_size
annealing_step = 10 * n_batches
lmb = 0.005
learning_rate = 0.1
keepProb = .5
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
    logLikeHood = torch.mean (torch.sum (p * torch.log(torch.sum (alpha, dim= 1, keepdims= True)) - \
                              torch.log (alpha), dim = 1, keepdims= True))

    #KL_reg =  tf.minimum(1.0, tf.cast(global_step/annealing_step, tf.float32)) * KL((alpha - 1)*(1-p) + 1 , K)
    KL_reg = min(1.0, float(global_step) / annealing_step) * \
                     KL((alpha - 1) * (1 - p) + 1 , numOfClass)
    return logLikeHood + KL_reg

# Three types evidence
def relu_evidence(logits):
    #return torch.nn.ReLU(logits)
    logits[logits < 0] = 0
    return logits

def exp_evidence(logits): 
    return torch.exp(logits / 1000)

def softmax_evidence(logits):
    return torch.nn.Softmax(logits)

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
        self.dropout = nn.Dropout(keepProb)

    def forward(self, input):
        out1 = self.maxPool(self.relu(self.conv1(input)))
        out2 = self.maxPool(self.relu(self.conv2(out1)))
        out3 = self.dropout(self.relu(self.fc1(out2.view(out2.size(0), -1))))
        out4 = self.fc2(out3)
        return out4

# Create ANN
model = CNNModel();
# SGD Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# CNNModel with softmax cross entropy loss function
list = []
lossList = []
for epoch in range(8):
    lossList.clear()
    print('Training-epoch: {}'.format(epoch + 1))
    for i, (images, labels) in enumerate(train_loader):
        train = images.view(100, 1, 28, 28)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train) 
        #print ('The grad_fn of outputs: {}'.format(outputs.requires_grad))
        # Calculate softmax and ross entropy loss
        # loss = error(outputs, labels)

        evidence = relu_evidence(outputs)
        #print ('The grad_fn of evidence: {}'.format(evidence.requires_grad))
        alpha = evidence + 1

        u = numOfClass / torch.sum(alpha, dim = 1, keepdims = True)
        pro = alpha / torch.sum (alpha, dim = 1, keepdims = True)
        #print ('The grad_fn of pro: {}'.format(pro.requires_grad))

        loss = torch.mean(loss_eq3(pro, alpha, numOfClass, global_step, annealing_step))
        l2Loss = (L2Loss(model.state_dict()['fc1.weight']) + 
                        L2Loss(model.state_dict()['fc2.weight'])) * lmb 
        #print ('The shape of loss: {}'.format(loss.shape))
        #print ('Uncertainty: {}'.format(u))
        
        lossList.append(loss)
        
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
    list.append(lossList)
    plt.plot(lossList, label= 'Epoch{}'.format(epoch + 1))
    plt.xlabel('Iters')
    plt.ylabel('LossVal')
plt.legend(loc= 0)
plt.show()

print ('Finish Training')
# save model
torch.save(model.state_dict(), '../data/model.pt')
# save testing dataset
np.savez('../data/test.npz', features_test, targets_test)
