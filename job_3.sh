#!/bin/bash
set -e


mkdir -p ./test_yeast5/result_MLP_normal_2000_normal/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_normal_2000 test_method=normal device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_normal_2000 test_method=normal device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_normal_2000 test_method=normal device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_normal_2000 test_method=normal device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_normal_2000 test_method=normal device_id=1



