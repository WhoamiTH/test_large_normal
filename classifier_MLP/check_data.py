# # -*- coding:utf-8 -*-

# import torch
# import torchvision
# import torchvision.transforms as transforms


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# train processing head
import sys

from sklearn.externals import joblib
from time import clock

import pandas as pd
import numpy as np
import sklearn.metrics as skmet
from sklearn.metrics import accuracy_score
# from pytorchtools import EarlyStopping
import torch
from torch import nn
from torch.nn import init
import time
import sampling



import numpy as np
import os
import pickle

def divide_data(Data, Label):
    print(Data.shape)
    print(Label.shape)

    positive_index = np.where(Label == 1)
    negative_index = np.where(Label == 0)

    positive = Data[positive_index[0]]
    negative = Data[negative_index[0]]
    return positive, negative


def save_data(file_name, data):
    fileObject = open(file_name, 'wb') 
    pickle.dump(data, fileObject)
    fileObject.close()


def load_data(file_name):
    fileObject = open(file_name, 'rb')
    modelInput = pickle.load(fileObject)
    fileObject.close()
    return modelInput

informative_minority_data_name = './informative_minority_data.pkl'
border_majority_data_name = './border_majority_data.pkl'

informative_minority_data = load_data(informative_minority_data_name)
border_majority_data = load_data(border_majority_data_name)


print(informative_minority_data.shape)
print(border_majority_data.shape)