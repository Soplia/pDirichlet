#### Graph for image rotation experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
import scipy.ndimage as nd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

# img: digit image (28, 28)
def rotating_image_classification(img, model, 
    numClass= 10, dims=(28, 28), threshold= 0.25, 
    c=['black','blue','brown','purple','cyan','red'] * 2, 
    marker=['s','^','o', '>'] * 4):

    outputsSave = np.zeros((1, 10))
    Mdeg = 180 
    Ndeg = Mdeg // 10 + 1
    ldeg = []
    lp = np.empty((1, numClass))
    lu = []
    rot_imgs = np.zeros((dims[0], dims[1] * Ndeg))

    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        rot_img = nd.rotate(img.reshape(*dims), deg, 
                            reshape= False).reshape(*dims)
        rot_img = np.clip(a= rot_img, a_min= 0, a_max= 1)
        rot_imgs[:, i * dims[1]: (i+1) * dims[1]] = 1 - rot_img
        outputs = model(torch.from_numpy(rot_img).type(torch.float).view(1, 1, 28, 28))
        outputs = outputs.detach()

        outputsSave = np.vstack((outputsSave, outputs))

        evidence = outputs * (outputs > 0)
        alpha = evidence + 1
        probability = alpha / torch.sum (alpha, dim = 1, keepdims = True)

        u = numClass / torch.sum(alpha, dim = 1, keepdims = True)
        lu.append(u)
        probability[probability < threshold] = 0
        lp = np.vstack((lp, probability.numpy()))
        ldeg.append(deg)

    outputsSave = outputsSave[1:, :]
    np.savez('../data/roatedDigitOutput.npz', outputsSave)

    plt.figure(figsize=[6,6])
    
    lp = lp[1:, :]
    
    for i in range(numClass):
        if lp[:, i].sum() != 0:
            plt.plot(ldeg, lp[:, i], label= '{}'.format(i),
                     marker= marker[i], c= c[i])

    lu = np.array(lu)       
    ldeg = np.array(ldeg)

    plt.plot(ldeg, lu, marker='<',c='red', label= 'uncertainty')
    plt.legend()
 
    plt.xlim([0, Mdeg])  
    plt.xlabel('Rotation Degree')
    plt.ylabel('Classification Probability')
    plt.show()

    #plt.figure(figsize=[6.4, 100])
    #plt.imshow(rot_imgs,cmap='gray')
    #plt.axis('off')
    #plt.show()