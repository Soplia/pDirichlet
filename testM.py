import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import rotateImg 

npzfile = np.load('../data/test.npz')

feaTh = torch.from_numpy(npzfile['arr_0']).type(torch.float)
tarTh = torch.from_numpy(npzfile['arr_1']).type(torch.int)

# batchSize = 100
# testDs = torch.utils.data.Dataset(feaTh, tarNp)
# testLd = torch.utils.data.DataLoader(testDs, batch_size= batchSize)

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

model = CNNModel()
model.load_state_dict(torch.load('../data/model.pt'))

outputs = model(feaTh.view(feaTh.shape[0], 1, 28, 28))
outputs = outputs.detach()
predictions = torch.argmax(outputs.data, dim= 1)
acc = (predictions == tarTh).sum() / float(predictions.shape[0])
print ('The acc of test dataset is {}'.format(100 * acc))

np.savez('../data/testM.npz', outputs.numpy(), tarTh.numpy())

########### RoatingImage ####################################
# digit = feaTh[4]
# # plt.imshow(digit.view((28, 28)))
# # plt.show()
# rotateImg.rotating_image_classification(digit, model)
# np.savez('../data/roatedDigitLabel.npz', tarTh[4])
# print ('Finish saving file!!!')



