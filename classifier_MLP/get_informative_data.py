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

train_data_name = 'all_train_data.pkl'
train_label_name = 'all_train_label.pkl'

test_data_name = 'all_test_data.pkl'
test_label_name = 'all_test_label.pkl'

train_data = load_data(train_data_name)
train_label = load_data(train_label_name)
test_data = load_data(test_data_name)
test_label = load_data(test_label_name)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

train_label = train_label.reshape(-1, 1)
test_label = test_label.reshape(-1, 1)

positive_data, negative_data = divide_data(train_data, train_label)
print('get pos neg data')

train_data_tras = train_data.reshape(train_data.shape[0], -1)
print('trans complete')


sampling_model = sampling.Sampling()
border_majority_index, informative_minority_index = sampling_model.getTrainingSample(train_data_tras, train_label)


informative_minority_data = negative_data[border_majority_index]
border_majority_data = positive_data[informative_minority_index]


informative_minority_data_name = './informative_minority_data.pkl'
border_majority_data_name = './border_majority_data.pkl'

save_data(informative_minority_data_name, informative_minority_data)
save_data(border_majority_data_name, border_majority_data)