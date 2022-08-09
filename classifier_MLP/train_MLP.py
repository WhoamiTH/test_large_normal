# train processing head
import sys

from sklearn.externals import joblib
from time import clock

import pandas as pd
import numpy as np
import sklearn.metrics as skmet
from sklearn.metrics import accuracy_score
from pytorchtools import EarlyStopping
import torch
from torch import nn
from torch.nn import init
import time
import sampling
import pickle


import os

def divide_data(Data, Label):
    positive_index = np.where(Label == 1)
    negative_index = np.where(Label == 0)

    positive = Data[positive_index[0]]
    negative = Data[negative_index[0]]
    return positive, negative

def generate_valid_data(data, label, size=0.05):
    # 按照比例划分训练集 和 validation 集合
    positive_data, negative_data = divide_data(data, label)
    positive_length = positive_data.shape[0]
    negative_length = negative_data.shape[0]
    positive_label = np.ones(positive_length).reshape(-1, 1)
    negative_label = np.zeros(negative_length).reshape(-1, 1)

    positive_data_label = np.hstack((positive_data, positive_label))
    negative_data_label = np.hstack((negative_data, negative_label))

    np.random.shuffle(positive_data_label)
    np.random.shuffle(negative_data_label)

    positive_valid_length = max(int(positive_length * size), 1)
    negative_valid_length = max(int(negative_length * size), 1)
    
    valid_pos_data_label = positive_data_label[:positive_valid_length, :]
    train_pos_data_label = positive_data_label[positive_valid_length:, :]

    valid_neg_data_label = negative_data_label[:negative_valid_length, :]
    train_neg_data_label = negative_data_label[negative_valid_length:, :]

    valid_data_label = np.vstack((valid_pos_data_label, valid_neg_data_label))
    train_data_label = np.vstack((train_pos_data_label, train_neg_data_label))

    np.random.shuffle(valid_data_label)
    np.random.shuffle(train_data_label)

    valid_data = valid_data_label[:, :-1]
    valid_label = valid_data_label[:, -1].reshape(-1, 1)

    train_data = train_data_label[:, :-1]
    train_label = train_data_label[:, -1].reshape(-1, 1)

    return valid_data, valid_label, train_data, train_label


def generate_batch_data(train_data, train_label, batch_size):
    data_length = train_data.shape[0]
    batch_size = min(data_length, batch_size)


    select_index = np.random.choice(data_length, batch_size, replace=False)
    select_data = train_data[select_index]
    select_label = train_label[select_index].reshape(-1, 1)


    return select_data, select_label



def loadTrainData(file_name):
    file_data = np.loadtxt(file_name, delimiter=',')
    label = file_data[:,-1]
    data = np.delete(file_data, -1, axis=1)
    data = data.astype(np.float64)
    label = label.reshape(-1, 1)
    label = label.astype(np.int)
    return data, label


def load_data(file_name):
    fileObject = open(file_name, 'rb')
    modelInput = pickle.load(fileObject)
    fileObject.close()
    return modelInput




def set_para():
    global dataset_name
    global dataset_index
    global record_index
    global device_id
    global train_method

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'dataset_name':
            dataset_name = para[1]
        if para[0] == 'dataset_index':
            dataset_index = para[1]
        if para[0] == 'record_index':
            record_index = para[1]
        if para[0] == 'device_id':
            device_id = para[1]
        if para[0] == 'train_method':
            train_method = para[1]




# -------------------------------------parameters----------------------------------------
dataset_name = 'abalone19'
dataset_index = '1'
record_index = '1'
device_id = '1'
method_name = 'MLP_normal'
train_method = 'MLP_bm_2000'
num_epochs = 5000
batch_size = 16
start_epochs = 0
save_epoch = 200
# ----------------------------------set parameters---------------------------------------
set_para()
train_file_name = '/srv/scratch/z5102138/cifar10/all_train_data.pkl'
train_label_name = '/srv/scratch/z5102138/cifar10/all_train_label.pkl'


model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, train_method, record_index, dataset_index)


os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


# ----------------------------------start processing-------------------------------------
print(train_file_name)
print(model_name)
print('----------------------\n\n\n')



start = time.process_time()


base_train_data = load_data(train_file_name)
base_train_label = load_data(train_label_name)
base_train_label = base_train_label.reshape(-1, 1)

base_train_data = base_train_data.transpose(0, 3, 1, 2)


model_type, infor_method, num_epochs = train_method.split('_')
num_epochs = int(num_epochs)


# positive_data, negative_data = divide_data(base_train_data, base_train_label)


input_dim = base_train_data.shape


patience = 20
# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)

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


# dependency_dict = {
#     2000 : 5000,
#     5000 : 8000,
#     8000 : 10000,
#     10000 : 15000,
#     15000 : 20000,
#     20000 : None
# }

dependency_dict = {
    200:500,
    500:1000,
    1000:1500,
    1500:2000,
    2000:None
}





# dependency_list = ['20000', '15000', '10000', '8000', '5000', '2000']
# dependency_list = ['2000', '1500', '1000', '500', '200']
dependency_list = ['200',  '500', '1000', '1500', '2000']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for train_epoch in dependency_list:
    history_train_method = 'MLP_{0}_{1}'.format(infor_method, train_epoch)
    history_model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, history_train_method, record_index, dataset_index)
    # 判断追训模型是否存在
    print(history_model_name)
    if os.path.exists(history_model_name):
        # 找到模型，追训
        net = torch.load(history_model_name, map_location=device)
        start_epochs = int(train_epoch)
    else:
        # 没有找到，开始训练
        break

if start_epochs == 0:
    net = Classification()
    init.normal_(net.fc1.weight, mean=0, std=0.01)
    init.constant_(net.fc1.bias, val=0)
    init.normal_(net.fc2.weight, mean=0, std=0.01)
    init.constant_(net.fc2.bias, val=0)
    init.normal_(net.fc3.weight, mean=0, std=0.01)
    init.constant_(net.fc3.bias, val=0)
    
    
net.to(device)

loss_fn = nn.BCELoss()
loss_fn_1 = nn.BCELoss(reduction='none')
loss_fn_2 = nn.BCELoss(reduction='none')
loss_fn_3 = nn.BCELoss(reduction='none')
loss_fn_4 = nn.BCELoss(reduction='none')

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(net.parameters(), lr=1.2, momentum=0.9)

# input_valid_pre_data = torch.Tensor(torch.from_numpy(transformed_valid_pre_data).float())
# input_valid_pos_data = torch.Tensor(torch.from_numpy(transformed_valid_pos_data).float())
# input_valid_label = torch.Tensor(torch.from_numpy(transformed_valid_label).float())
# input_valid_pre_data = input_valid_pre_data.to(device)
# input_valid_pos_data = input_valid_pos_data.to(device)
# input_valid_label = input_valid_label.to(device)



# input_valid_data = torch.Tensor(torch.from_numpy(valid_data).float())
# input_valid_label = torch.Tensor(torch.from_numpy(valid_label).float())
# input_valid_data = input_valid_data.to(device)
# input_valid_label_gpu = input_valid_label.to(device)


for epoch in range(start_epochs+1, num_epochs+1):
    train_x, train_y = generate_batch_data(positive_data, negative_data, infor_method, informative_minority_data, border_majority_data, batch_size)


    input_data = torch.Tensor(torch.from_numpy(train_x).float())
    input_label = torch.Tensor(torch.from_numpy(train_y).float())

    input_data = input_data.to(device)
    input_label = input_label.to(device)
    
    predict = net(input_data)

    loss = loss_fn(predict, input_label)
    # loss_1 = loss_fn_1(predict[:, 0], input_label[:,0])
    # loss_2 = loss_fn_2(predict[:, 1], input_label[:,1])
    # loss_3 = loss_fn_3(predict[:, 2], input_label[:,2])
    # loss_4 = loss_fn_4(predict[:, 3], input_label[:,3])
    # loss = torch.sum( loss_1 + loss_2 + loss_3 + loss_4 )

    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss = loss.item()

    if epoch % 100 == 0:
        # valid_output = net(input_valid_data)
        # result =  torch.ge(valid_output, 0.5) 
        # result = result.cpu()
        # #计算准确率
        # train_acc = accuracy_score(input_valid_label, result)

        # #计算精确率
        # pre = skmet.precision_score(y_true=input_valid_label, y_pred=result)

        # #计算召回率
        # rec = skmet.recall_score(y_true=input_valid_label, y_pred=result)
        # f1 = skmet.f1_score(y_true=input_valid_label, y_pred=result)
        # auc = skmet.roc_auc_score(y_true=input_valid_label, y_score=result)
        # print('epoch {:.0f}, loss {:.4f}, train acc {:.2f}%, f1 {:.4f}, precision {:.4f}, recall {:.4f}, auc {:.4f}'.format(epoch+1, train_loss, train_acc*100, f1, pre, rec, auc) )
        print('epoch {:.0f}, loss {:.4f}'.format(epoch+1, train_loss) )
    if cur_train_epochs == save_epoch:
        cur_train_method_list = [model_type, infor_method, str(save_epoch)]
        cur_train_method = '_'.join(cur_train_method_list)
        cur_model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, cur_train_method, record_index, dataset_index)
        torch.save(net, cur_model_name)
        save_epoch = dependency_dict[epoch]
        print('save model {0}'.format(cur_model_name))

        last_train_epoch = 0
        for train_epoch in dependency_list:
            history_train_method = 'MLP_{0}_{1}'.format(infor_method, train_epoch)
            history_model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, history_train_method, record_index, dataset_index)
            # 判断追训模型是否存在
            print(history_model_name)
            if os.path.exists(history_model_name):
                # 更新已有模型标记
                last_train_epoch = train_epoch
            else:
                save_epoch = train_epoch
                if last_train_epoch != cur_train_epochs:
                    history_train_method = 'MLP_{0}_{1}'.format(infor_method, last_train_epoch)
                    history_model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, history_train_method, record_index, dataset_index)
                    net = torch.load(history_model_name, map_location=device)
                    cur_train_epochs = last_train_epoch
                break
    
    if cur_train_epochs >= num_epochs:
        break

            
cur_train_method_list = [model_type, infor_method, str(save_epoch)]
cur_train_method = '_'.join(cur_train_method_list)
cur_model_name = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset_name, cur_train_method, record_index, dataset_index)
torch.save(net, cur_model_name)
print('save model {0}'.format(cur_model_name))   

torch.save(net, model_name)


finish = time.process_time()
running_time = finish-start
print('running_time is {0}'.format(running_time))

