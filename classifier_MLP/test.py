import sys
from sklearn.externals import joblib
from time import clock
import handle_data
import predict_test
import numpy as np
import csv
import re
import handle_data
import predict_test
import pandas as pd
import numpy as np

import sklearn.metrics as skmet
# import tensorflow as tf

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import torch
from torch import nn
from torch.nn import init
from sklearn.metrics import confusion_matrix
import sampling
import os
import pickle


def get_test_ref_samples(test_data, reference_data, reference_label, test_ref_num):
    length_test_data = test_data.shape[0]
    length_reference_data = reference_data.shape[0]

    # repeat 每一个都连续重复
    test_samples = np.repeat(test_data, test_ref_num, axis=0)
    # reference data
    ref_index_list = []
    for epoch in range(length_test_data):
        ref_index_list.append( np.random.choice(length_reference_data, test_ref_num, replace=False) )
    ref_index = np.array(ref_index_list).flatten()

    ref_samples = reference_data[ref_index]
    ref_label = reference_label[ref_index]
    return test_samples, ref_samples, ref_label


def transform_data_to_test_form(test_data, reference_data, reference_label, test_ref_num):
    test_samples, ref_samples, ref_label = get_test_ref_samples(test_data, reference_data, reference_label, test_ref_num)
    # transformed_test_data = np.hstack((test_samples, ref_samples))
    transformed_test_data = np.concatenate((test_samples, ref_samples), axis=1)
    ref_label = ref_label.reshape(-1, 1)
    return transformed_test_data, ref_label




def divide_data(Data, Label):
    positive_index = np.where(Label == 1)
    negative_index = np.where(Label == 0)

    positive = Data[positive_index[0]]
    negative = Data[negative_index[0]]

    return positive, negative


def loadTrainData(file_name):
    file_data = np.loadtxt(file_name, delimiter=',')
    label = file_data[:,-1]
    data = np.delete(file_data, -1, axis=1)
    data = data.astype(np.float64)
    label = label.reshape(-1, 1)
    label = label.astype(np.int)
    return data, label



def get_test_ref_num(dataset, ref_num_type, ref_times):
    ref_times = int(ref_times)
    if ref_num_type == 'num':
        return ref_times
    if ref_num_type == 'IR':
        dataset_dict = {
            'pima': 2,
            'glass0': 3,
            'vehicle0': 4,
            'ecoli1': 4,
            'yeast3': 9,
            'pageblocks1': 16,
            'glass5': 23,
            'yeast5': 33,
            'yeast6': 40,
            'abalone19': 129
        }
        basic_num = dataset_dict[dataset]
        return basic_num * ref_times

def get_infor_data(train_data, train_label):
    sampling_model = sampling.Sampling()
    border_majority_index, informative_minority_index = sampling_model.getTrainingSample(train_data, train_label)
    informative_minority_data = negative_data[border_majority_index]
    border_majority_data = positive_data[informative_minority_index]
    return informative_minority_data, border_majority_data


def get_reference_data(train_data, positive_data, negative_data, informative_minority_data, border_majority_data,  ref_data_type):
    if ref_data_type == 'normal':
        ref_label = np.zeros(negative_data.shape[0]).reshape(-1, 1)
        return negative_data, ref_label
    
    if ref_data_type == 'pos':
        ref_label = np.ones(positive_data.shape[0]).reshape(-1, 1)
        return positive_data, ref_label
    
    if ref_data_type == 'neg':
        ref_label = np.zeros(negative_data.shape[0]).reshape(-1, 1)
        return negative_data, ref_label
    
    if ref_data_type == 'im':
        ref_label = np.ones(informative_minority_data.shape[0]).reshape(-1, 1)
        return informative_minority_data, ref_label

    if ref_data_type == 'bm':
        ref_label = np.zeros(border_majority_data.shape[0]).reshape(-1, 1)
        return border_majority_data, ref_label

    if ref_data_type == 'both':
        pos_ref_label = np.ones(informative_minority_data.shape[0]).reshape(-1, 1)
        neg_ref_label = np.zeros(border_majority_data.shape[0]).reshape(-1, 1)
        # pos_ref_data = np.hstack((informative_minority_data, pos_ref_label))
        # neg_ref_data = np.hstack((border_majority_data, neg_ref_label))

        ref_data = np.concatenate((informative_minority_data, border_majority_data), axis=0)
        ref_label = np.vstack((pos_ref_label, neg_ref_label))
        return ref_data, ref_label

def get_boundray_num(test_ref_num, boundary_type):
    if boundary_type == 'half':
        return int(test_ref_num/2)
    else:
        return int(boundary_type)

def load_data(file_name):
    fileObject = open(file_name, 'rb')
    modelInput = pickle.load(fileObject)
    fileObject.close()
    return modelInput



# def get_test_info(test_method):
#     test_info_list = test_method.split('_')
    
#     if len(test_info_list) == 1:
#         transform_method = test_info_list[0]
#         ref_data_type = 'random'
#         ref_num_type = 'num'
#         ref_times = '1'
#         boundary_type = '1'
    
#     if len(test_info_list) == 4:
#         ref_data_type, ref_num_type, ref_times, boundary_type = test_info_list

#     return transform_method, ref_data_type, ref_num_type, ref_times, boundary_type
    
def set_para():
    global dataset_name
    global dataset_index
    global record_index
    global train_method
    global test_method
    global device_id

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'dataset_name':
            dataset_name = para[1]
        if para[0] == 'dataset_index':
            dataset_index = para[1]
        if para[0] == 'record_index':
            record_index = para[1]
        if para[0] == 'train_method':
            train_method = para[1]
        if para[0] == 'test_method':
            test_method = para[1]
        if para[0] == 'device_id':
            device_id = para[1]

# -------------------------------------parameters----------------------------------------
dataset_name = 'abalone19'
dataset_index = '1'
record_index = '1'
train_method = 'MLP_minus_mirror'
test_method = 'minus_random_num_10_half'
device_id = '1'
# ----------------------------------set parameters---------------------------------------
set_para()

os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

# train_file_name = '/srv/scratch/z5102138/cifar10/all_train_data.pkl'
# train_label_name = '/srv/scratch/z5102138/cifar10/all_train_label.pkl'

# test_file_name = '/srv/scratch/z5102138/cifar10/all_test_data.pkl'
# test_label_name = '/srv/scratch/z5102138/cifar10/all_test_label.pkl'

train_file_name = '/data/hantai/cifar10/all_train_data.pkl'
train_label_name = '/data/hantai/cifar10/all_train_label.pkl'

test_file_name = '/data/hantai/cifar10/all_test_data.pkl'
test_label_name = '/data/hantai/cifar10/all_test_label.pkl'


# ############################# 注意修改模型及方法名称 ##############################
model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, train_method, record_index, dataset_index)
result_record_path = './test_{0}/result_{1}_{2}/record_{3}/'.format(dataset_name, train_method, test_method, record_index)
record_name_prefix = '{0}_{1}_pred_result'.format(dataset_name, dataset_index)

print(train_file_name)
print(test_file_name)
print(train_method)
print(test_method)
print(model_name)
print(result_record_path)
print('----------------------\n\n\n')

# # 在测试方法中，用 _ 分割了不同的测试部分
# ref_data_type, ref_num_type, ref_times, boundary_type = test_method.split('_')



# ------------- load train data and find reference data --------------------------------
# train_data, train_label = loadTrainData(train_file_name)


# train_data = load_data(train_file_name)
# train_label = load_data(train_label_name)
# train_label = train_label.reshape(-1, 1)
# train_data = train_data.transpose(0, 3, 1, 2)

test_data = load_data(test_file_name)
test_label = load_data(test_label_name)
test_label = test_label.reshape(-1, 1)
test_label = np.where(test_label!=3, 0, 1)
test_data = test_data.transpose(0, 3, 1, 2)

# positive_data, negative_data = divide_data(train_data, train_label)

# informative_minority_data = load_data(informative_minority_data_name)
# border_majority_data = load_data(border_majority_data_name)

# informative_minority_data = informative_minority_data.transpose(0, 3, 1, 2)
# border_majority_data = border_majority_data.transpose(0, 3, 1, 2)

# informative_minority_data, border_majority_data = get_infor_data(train_data, train_label)
# reference_data, reference_label = get_reference_data(train_data, positive_data, negative_data, informative_minority_data, border_majority_data, ref_data_type)

# if ref_times == 'all':
#     test_ref_num = reference_data.shape[0]
# else:
#     test_ref_num = get_test_ref_num(dataset_name, ref_num_type, ref_times)
#     # # 如果 test_ref_num 超了 换成所有的 reference 数据
#     test_ref_num = min(test_ref_num, reference_data.shape[0])
# boundary_number = get_boundray_num(test_ref_num, boundary_type)

# test_data, test_label = loadTrainData(test_file_name)
# cur_test_data, ref_label = transform_data_to_test_form(test_data, reference_data, reference_label, test_ref_num)



start = clock()

class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.max_pool2d = nn.functional.max_pool2d()

        self.conv1 = nn.Conv2d(3, 6, 16) 
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1   = nn.Linear(64, 32)  
        self.fc2   = nn.Linear(32, 16)
        self.fc3   = nn.Linear(16, 1)
  
    def forward(self, x1, x2): 
        x = self.conv1(x1)
        x = self.relu(x)
        x = nn.functional.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = self.relu(x)
        x = nn.functional.max_pool2d(x, 2) 
        x = x.view(x.size()[0], -1)  # 展平  x.size()[0]是batch size
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)   
        return x

net = torch.load(model_name)

def get_MLP_predict_result(test_data, net):
    test_input_data = torch.Tensor(torch.from_numpy(test_data).float())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_input_data = test_input_data.to(device)

    output = net(test_input_data_1, test_input_data_2)
    return output

cur_test_data = test_data
# 获取初步结果
test_result = get_MLP_predict_result(cur_test_data, net)
test_result = test_result.detach().to('cpu').numpy()
test_result = test_result[:, 0]

test_length = test_data.shape[0]
# test_pred_results = test_result.reshape(test_length, test_ref_num)
# test_pred_results[test_pred_results<0.5] = 0
# test_pred_results[test_pred_results>=0.5] = 1
# test_pred_results = test_result

# 获取 vote 结果
# if transform_method == 'normal':
#     general_vote_results = test_pred_results
# else:
# general_vote_results = np.sum(test_pred_results, axis=1)
# general_vote_results[general_vote_results<boundary_number] = 0
# general_vote_results[general_vote_results>=boundary_number] = 1

# general_vote_results = general_vote_results.reshape(-1,1)
general_vote_results = test_result


# 保存测试及预估结果数据
# test_predict_data_result = np.hstack((test_data, general_vote_results))
np.savetxt(result_record_path+record_name_prefix+'.csv', general_vote_results, delimiter=',')

# 计算预估效果
true_label = test_label
predict_label = general_vote_results

precision = skmet.precision_score(y_true=true_label, y_pred=predict_label)
recall = skmet.recall_score(y_true=true_label, y_pred=predict_label)
fscore = skmet.f1_score(y_true=true_label, y_pred=predict_label)
auc = skmet.roc_auc_score(y_true=true_label, y_score=predict_label)

print("the AUC is {0}\n".format(auc))
print("the Fscore is {0}\n".format(fscore))
print("the precision is {0}\n".format(precision))
print("the recall is {0}\n".format(recall))

with open(result_record_path+record_name_prefix+'.txt','w') as record:
    record.write("the AUC is {0}\n".format(auc))
    record.write("the Fscore is {0}\n".format(fscore))
    record.write("the precision is {0}\n".format(precision))
    record.write("the recall is {0}\n".format(recall))

print('Done')