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
    positive_index = np.where(Label == 1)
    negative_index = np.where(Label == 0)

    positive = Data[positive_index[0]]
    negative = Data[negative_index[0]]
    return positive, negative

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding='latin1') # dict类型
    X = datadict['data']        # X, ndarray, 像素值
    Y = datadict['labels']      # Y, list, 标签, 分类
    
    # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
    # transpose，转置
    # astype，复制，同时指定类型
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = [] # list
  ys = []
  
  # 训练集batch 1～5
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X) # 在list尾部添加对象X, x = [..., [X]]
    ys.append(Y)  
  
  Xtr = np.concatenate(xs) # [ndarray, ndarray] 合并为一个ndarray
  Ytr = np.concatenate(ys)
  Ytr = np.where(Ytr!=3, 0, 1)
  del X, Y

  # 测试集
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

root = '/data/hantai/cifar10/cifar-10-batches-py'

train_data, train_label, test_data, test_label = load_CIFAR10(root)
train_label = train_label.reshape(-1, 1)
test_label = test_label.reshape(-1, 1)


print(train_data.shape)
print(train_label.shape)


def save_data(file_name, data):
    fileObject = open(file_name, 'wb') 
    pickle.dump(data, fileObject)
    fileObject.close()

train_data_name = 'all_train_data.pkl'
train_label_name = 'all_train_label.pkl'

test_data_name = 'all_test_data.pkl'
test_label_name = 'all_test_label.pkl'

save_data(train_data_name, train_data)
save_data(train_label_name, train_label)


print(train_data.shape)
print(train_label.shape)

save_data(test_data_name, test_data)
save_data(test_label_name, test_label)


print(test_data.shape)
print(test_label.shape)

# train_data_tras = train_data.reshape(train_data.shape[0], -1)


# sampling_model = sampling.Sampling()
# border_majority_index, informative_minority_index = sampling_model.getTrainingSample(train_data_tras, train_label)

# print(border_majority_index)
# print(informative_minority_index)