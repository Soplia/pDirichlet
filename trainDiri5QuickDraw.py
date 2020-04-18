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
from Modelj import * 

npzfile = np.load('./Dataset/train.npz')
features_numpy = npzfile['data'] / 255.0
targets_numpy = npzfile['target'] 

numOfClass = 5

targets_train = targets_numpy[targets_numpy < numOfClass]
features_train = features_numpy[targets_numpy < numOfClass]

#idx = np.random.randint(0, len(features_train))
#plt.imshow(features_train[idx].reshape(28,28)) 
#plt.show()

featuresTrain = torch.from_numpy(features_train).type(torch.FloatTensor)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) 

epochs = 9
batch_size = 256
annealing_step = 10 * (len(featuresTrain) // batch_size)
lmb = 0.005
learning_rate = 0.1

print(f'{epochs} epochs in training')

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
# Data loader
train_loader = torch.utils.data.DataLoader(train, 
                                                                     batch_size = batch_size, 
                                                                     shuffle = True)

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

model = CNNModel(numOfClass)
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

# CNNModel with softmax cross entropy loss function
acc1d = []
loss1d = []
fig, axes = plt.subplots(nrows= 2, ncols= 1)

for epoch in range(epochs):
    global_step = 10
    print('Training-epoch: {}'.format(epoch + 1))
    for i, (images, labels) in enumerate(train_loader):
        # Change the shape of labels from (images.shape[0]) to 
        # (images.shape[0], 10)
        newLabels = torch.zeros((labels.shape[0], numOfClass), dtype= torch.float32)
        cnt = 0
        for pos in labels:
            newLabels[cnt, pos] = 1.0
            cnt += 1

        train = images.view(-1, 1, 28, 28)
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

        (loss + l2Loss).backward()
        optimizer.step()
        global_step += 1
print ('Finish Training')

# save model
torch.save(model.state_dict(), './criticalData/model5DiriQuickdraw{}.pt'.format(epochs))
print ('Finish Saving Files')

axes[0].plot(acc1d, label= 'Accuracy')
axes[1].plot(loss1d, label= 'Loss')
axes[0].set_ylabel('AccVal')
axes[1].set_ylabel('LossVal')
axes[1].set_xlabel('Iteration')
axes[0].legend()
axes[1].legend() 
plt.show()