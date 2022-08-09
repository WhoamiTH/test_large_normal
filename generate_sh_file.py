
# # -*- coding:utf-8 -*-



# ---------------------  分割线 在此下方添加数据 -----------------------------------



# ---------------------  检查 test -----------------------------------
import sys
import os
import math
# dataset_list = ['abalone19', 'ecoli1', 'glass0', 'glass5', 'pageblocks1', 'pima', 'vehicle0', 'yeast3', 'yeast5', 'yeast6']
dataset_list = ['yeast5']
data_range = 5
record_index = 1

# train_infor_method_list = ['normal', 'bm', 'im', 'im2', 'im3', 'both', 'both2', 'both3']
train_infor_method_list = ['normal']
# early_stop_type_list = ['2000', '5000', '8000', '10000', '15000', '20000']
early_stop_type_list = ['200', '500', '1000', '1500', '2000']


# test_infor_method_list = [ 'normal', 'bm', 'im', 'both']
test_infor_method_list = [ 'normal']

# ref_num_type_list = ['num']
# ref_times_list = ['10']
# boundary_type_list = ['half']


device_id = 0
train_method = ''
test_method = ''
train_command_list = []
test_command_list = []
train_num = 0
test_num = 0
for dataset in dataset_list:
    for sample_method in train_infor_method_list:
        cur_command_list = []
        cur_valid_command_list = []
        cur_train_num = 0
        cur_test_num = 0

        # 创建训练结果目录
        cur_train_dir_list = []
        for early_stop_type in early_stop_type_list:
            train_method = 'MLP_{0}_{1}'.format(sample_method, early_stop_type)
            train_dir_com_str = 'mkdir -p ./test_{0}/model_{2}/record_{1}/\n'.format(dataset, record_index, train_method)
            cur_train_dir_list.append(train_dir_com_str)
        cur_train_dir_list.append('\n\n\n')

        # 根据训练模型，创建训练任务
        cur_train_com_list = []
        for dataset_index in range(1, 6):
            cur_path = './test_{0}/model_{1}/record_{2}/{1}_{3}'.format(dataset, train_method, record_index, dataset_index)
            if not os.path.exists(cur_path):
                trian_com_str = 'python3 ./classifier_MLP/train_MLP.py dataset_name={0} dataset_index={1} record_index=1 device_id={2} train_method={3}\n'.format(dataset, dataset_index, device_id, train_method)
                cur_train_com_list.append(trian_com_str)
                # cur_valid_command_list.append(trian_com_str)
                cur_train_num += 1
        if len(cur_train_com_list) > 0:
            cur_train_com_list.append('\n\n\n')
        
        cur_test_com_list = []
        for test_infor_method in test_infor_method_list:
            # for ref_num_type in ref_num_type_list:
            #     for ref_times in ref_times_list:
            #         for boundary_type in boundary_type_list:
            # test_method = '{0}_{1}_{2}_{3}'.format(test_infor_method, ref_num_type, ref_times, boundary_type)
            test_method = test_infor_method
            
            for early_stop_type in early_stop_type_list:
                cur_test_com_sub_list = []
                train_method = 'MLP_{0}_{1}'.format(sample_method, early_stop_type)
                mkdir_command = 'mkdir -p ./test_{0}/result_{1}_{2}/record_{3}/\n'.format(dataset, train_method, test_method, record_index)
                test_path_flag = False
                for dataset_index in range(1, 6):
                    cur_path = './test_{0}/result_{1}_{2}/record_{3}/{0}_{4}_pred_result.txt'.format(dataset, train_method, test_method, record_index, dataset_index)
                    if not os.path.exists(cur_path):
                        if not test_path_flag:
                            test_path_flag = True
                            cur_test_com_sub_list.append(mkdir_command)
                        test_com_str = 'python3 ./classifier_MLP/test.py dataset_name={0} dataset_index={1} record_index=1 train_method={2} test_method={3} device_id={4}\n'.format(dataset, dataset_index, train_method, test_method, device_id)
                        cur_test_com_sub_list.append(test_com_str)
                        # cur_valid_command_list.append(test_com_str)
                        cur_test_num += 1
                if test_path_flag:
                    cur_test_com_sub_list.append('\n\n\n')
                if len(cur_test_com_sub_list) > 0:
                    cur_test_com_list.append(cur_test_com_sub_list)


        # if len(cur_train_com_list) > 0:
        #     train_command_list.append(cur_train_dir_list)
        #     train_command_list.append(cur_train_com_list)
            # cur_dataset_com_list += cur_test_com_list
            # command_list.append(cur_dataset_com_list)
        if len(cur_test_com_list) > 0:
            test_command_list += cur_test_com_list
        train_num += len(cur_train_com_list)
        test_num += len(cur_test_com_list)

print('train_num is {0}'.format(train_num))
print('train com num is {0}'.format(len(train_command_list)))
print('test_num is {0}'.format(test_num))
print('test com num is {0}'.format(len(test_command_list)))

total_file_num = 4
# total_length = len(command_list)
train_start = 0
test_start = 0
train_offset = math.ceil(float(len(train_command_list))/total_file_num)
test_offset = math.ceil(float(len(test_command_list))/total_file_num)
# print(total_length)
# print(offset)
for file_index in range(1, total_file_num+1):
    # print(file_index)
    # print(start, offset)
    if file_index < total_file_num:
        
        cur_train_command_list = train_command_list[train_start:train_start+train_offset]
        cur_test_command_list = test_command_list[test_start:test_start+test_offset]
        train_start += train_offset
        test_start += test_offset
    else:
        cur_train_command_list = train_command_list[train_start:]
        cur_test_command_list = test_command_list[test_start:]
    cur_command_list = cur_train_command_list + cur_test_command_list
    print(len(cur_command_list))
    
    with open('job_{0}.sh'.format(file_index), 'w') as fsh:
        fsh.write('#!/bin/bash\n')
        fsh.write('set -e\n\n\n')
        for item_command_list in cur_command_list:
            for line in item_command_list:
                if isinstance(line, str):
                    fsh.write(line)
                if isinstance(line, list):
                    for sub_line in line:
                        fsh.write(sub_line)
    
    
    
    
    # with open('job_{0}.qjob'.format(file_index), 'w') as fsh:
    #     fsh.write('# 选择资源\n\n\n')
    #     fsh.write('#PBS -N test_ln\n')
    #     fsh.write('#PBS -l ngpus=1\n')
    #     fsh.write('#PBS -l mem=46gb\n')
    #     fsh.write('#PBS -l ncpus=8\n')
    #     fsh.write('#PBS -l walltime=12:00:00\n')
    #     fsh.write('#PBS -M han.tai@student.unsw.edu.au\n')
    #     fsh.write('#PBS -m ae\n')
    #     fsh.write('#PBS -j oe\n\n')
    #     fsh.write('#PBS -o /srv/scratch/z5102138/test_large_normal/\n')
    #     fsh.write('source ~/anaconda3/etc/profile.d/conda.sh\n')
    #     fsh.write('conda activate py36\n\n\n')
    #     fsh.write('cd /srv/scratch/z5102138/test_large_normal\n')
    #     fsh.write('which python\n\n\n\n')
    #     for item_command_list in cur_command_list:
    #         for line in item_command_list:
    #             if isinstance(line, str):
    #                 fsh.write(line)
    #             if isinstance(line, list):
    #                 for sub_line in line:
    #                     fsh.write(sub_line)

