import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
print(accuracy_score(y_true, y_pred))  # 0.5
print(accuracy_score(y_true, y_pred, normalize=False)) 

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print(precision_score(y_true, y_pred, average='macro'))  # 0.2222222222222222
print(precision_score(y_true, y_pred, average='micro'))  # 0.3333333333333333
print(precision_score(y_true, y_pred, average='weighted'))  # 0.2222222222222222
print(precision_score(y_true, y_pred, average=None))  # [0.66666667 0.         0.        ]

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print(recall_score(y_true, y_pred, average='macro'))  # 0.3333333333333333
print(recall_score(y_true, y_pred, average='micro'))  # 0.3333333333333333
print(recall_score(y_true, y_pred, average='weighted'))  # 0.3333333333333333
print(recall_score(y_true, y_pred, average=None))  # [1. 0. 0.]