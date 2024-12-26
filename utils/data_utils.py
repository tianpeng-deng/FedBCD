# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import torch
from utils.ultrasound_video_utils import Dataset_Video
from utils.ultrasound_image_utils import Custom_Dataset, set_transform

client_dict = {0: 'TDSC', 1: 'BUSI', 2: 'GDPH', 3: 'SYSUCC'}

def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data_ultrasound(dataset, client_name, fold, is_train=True):
    data_root = './dataset/'
    data_path = os.path.join(data_root, 'data', client_name)
    img_size = 160
    if client_name != 'TDSC':
        if is_train:
            trainset_path = os.path.join(data_root, fold, client_name, 'train.csv')
            train_transform = set_transform(img_size, ["gray","randomHflip", "totensor"])
            trainset = Custom_Dataset(data_path, trainset_path, train_transform)
            X_train = torch.empty(len(trainset), *trainset[0][0].shape).type(torch.float32)
            y_train = torch.empty(len(trainset), *trainset[0][1].shape).type(torch.int64)
            for i, (x, y) in enumerate(trainset):
                X_train[i] = x.type(torch.float32)
                y_train[i] = torch.from_numpy(y).type(torch.int64)
            train_data = [(x, y) for x, y in zip(X_train, y_train)]
            return train_data
        else:
            testset_path =  os.path.join(data_root, fold, client_name, 'test.csv')
            test_transform = set_transform(img_size, ["gray","totensor"])
            testset = Custom_Dataset(data_path, testset_path, test_transform)
            X_test = torch.empty(len(testset), *testset[0][0].shape).type(torch.float32)
            y_test = torch.empty(len(testset), *testset[0][1].shape).type(torch.int64)
            for i, (x, y) in enumerate(testset):
                X_test[i] = x.type(torch.float32)
                y_test[i] = torch.from_numpy(y).type(torch.int64)
            test_data = [(x, y) for x, y in zip(X_test, y_test)]
            return test_data
    if client_name == 'TDSC':
        if is_train:
            trainset_path = os.path.join(data_root, fold, client_name, 'train.csv')
            trainset = Dataset_Video(data_path, trainset_path, [img_size, img_size, 24])
            X_train = torch.empty(len(trainset), *trainset[0][0].shape).type(torch.float32)
            y_train = torch.empty(len(trainset), *trainset[0][1].shape).type(torch.int64)
            for i, (x, y) in enumerate(trainset):
                X_train[i] = torch.from_numpy(x).type(torch.float32)
                y_train[i] = torch.from_numpy(y).type(torch.int64)
            train_data = [(x, y) for x, y in zip(X_train, y_train)]
            return train_data
        else:
            testset_path =  os.path.join(data_root, fold, client_name, 'test.csv')
            testset = Dataset_Video(data_path, testset_path, [img_size, img_size, 24])
            X_test = torch.empty(len(testset), *testset[0][0].shape).type(torch.float32)
            y_test = torch.empty(len(testset), *testset[0][1].shape).type(torch.int64)
            for i, (x, y) in enumerate(testset):
                X_test[i] = torch.from_numpy(x).type(torch.float32)
                y_test[i] = torch.from_numpy(y).type(torch.int64)
            test_data = [(x, y) for x, y in zip(X_test, y_test)]
            return test_data

        
def read_client_data(dataset, idx, fold, is_train=True):
    if "US" in dataset:
        if isinstance(idx, int):
            return read_client_data_ultrasound(dataset, client_dict[idx], fold, is_train)
    
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
