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
    c=['black','blue','brown','purple','cyan','red'], 
    marker=['s','^','o'] * 2):

    Mdeg = 180 
    Ndeg = Mdeg // 10 + 1
    ldeg = []
    lp = []
    lu = []
    # scores = []
    scores = torch.zeros((1, numClass))
    rot_imgs = np.zeros((dims[0], dims[1] * Ndeg))

    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        rot_img = nd.rotate(img.reshape(*dims), deg, 
                            reshape= False).reshape(*dims)
        rot_img = np.clip(a= rot_img, a_min= 0, a_max= 1)
        rot_imgs[:, i * dims[1]: (i+1) * dims[1]] = 1 - rot_img
        outputs = model(torch.from_numpy(rot_img).type(torch.float).view(1, 1, 28, 28))
        p_pred_t = torch.argmax(outputs)
        evidence = outputs * (outputs > 0)
        alpha = evidence + 1

        u = numClass / torch.sum(alpha, dim = 1, keepdims = True)
        lu.append(u)

        # feed_dict={X: rot_img.reshape(1,-1), keep_prob: 1.0}
        # if uncertainty is None:
        #     p_pred_t = sess.run(prob, feed_dict= feed_dict)
        # else:
        #     p_pred_t,u = sess.run([prob,uncertainty], feed_dict=feed_dict)
        #     lu.append(u.mean())

        scores += p_pred_t >= threshold
        # scores.append((p_pred_t >= threshold).item())
        print (scores)

        ldeg.append(deg) 
        lp.append(p_pred_t.item())
        
    labels = np.arange(numClass)[scores[0].type(bool)]
    lp = np.array(lp)[:, labels]
    labels = labels.tolist()
    
    plt.figure(figsize=[6,6])
    for i in range(len(labels)):
        plt.plot(ldeg, lp[:,i], marker= marker[i], c= c[i])
    
    labels += ['uncertainty']
    plt.plot(ldeg, lu, marker='<',c='red')
        
    plt.legend(labels)
 
    plt.xlim([0, Mdeg])  
    plt.xlabel('Rotation Degree')
    plt.ylabel('Classification Probability')
    plt.show()

    plt.figure(figsize=[6.4, 100])
    plt.imshow(rot_imgs,cmap='gray')
    plt.axis('off')
    plt.show()