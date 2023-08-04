import torch
from torch.utils.data.dataset import T_co
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import numpy as np
import pandas as pd
import scipy.io as io
import random
import math

from config import Constant
from plot.PLOT import plot_split_distribution, plot_curve


class SpectralDataset(Dataset):
    def __init__(self, X, y, mat, indexs):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

        # Mat is not used in the actual code, only an interface is left here
        self.mat = mat.astype(np.float32) if mat is not None else mat
        self.indexs = indexs  # sample index

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        if self.mat is None:
            # return self.X[index], self.y[index], self.indexs[index]
            # Set it to self.indexs[0] here first, and change it later if necessary
            return self.X[index], self.y[index], self.indexs[0]
        else:
            return self.X[index], self.y[index], self.mat[index], self.indexs[index]

    def get_y(self, indexs=None):
        if indexs is None:
            return self.y
        else:
            return self.y[indexs]

    def get_x(self):
        return self.X

    def get_indexs(self):
        return self.indexs


def read_single_csv_data(csv_path):
    data = pd.read_csv(csv_path, header=0)
    y_brix = data['Brix'].values
    y_hardness = data['Hardness'].values
    X = data.drop(['Brix', 'Hardness'], axis=1).values
    return X, y_brix, y_hardness


def read_single_mat_data(mat_path):
    mat = io.loadmat(mat_path)
    return mat['name']


def read_data(csv_path_list, mat_path_list):
    dataset_num = len(csv_path_list)

    # Set not to read mat temporarily
    flag = False

    X = None
    y_brix = None
    y_hardness = None
    mat = None

    for i in range(dataset_num):
        csv_path = csv_path_list[i]
        mat_path = mat_path_list[i]
        X_item, y_brix_item, y_hardness_item = read_single_csv_data(csv_path)
        if flag: mat_item = read_single_mat_data(mat_path)
        print('dataset ', i, ' number of samples ', X_item.shape[0])
        if i == 0:
            X = X_item
            y_brix = y_brix_item
            y_hardness = y_hardness_item
            if flag: mat = mat_item
        else:
            X = np.vstack((X, X_item))
            y_brix = np.hstack((y_brix, y_brix_item))
            y_hardness = np.hstack((y_hardness, y_hardness_item))
            if flag: mat = np.vstack((mat, mat_item))

    return X, y_brix, y_hardness, mat


def split_train_test_data(csv_path, train_split_rate, property):
    # Single dataset, divided into training and testing sets, return matrix
    X, y_brix, y_hardness = read_single_csv_data(csv_path)

    # Generate subscripts and shuffle them
    indices = [i for i in range(X.shape[0])]
    np.random.shuffle(indices)

    split = int(np.floor((1 - train_split_rate) * len(indices)))
    # Randomly partitioned index
    train_indices, test_indices = indices[split:], indices[:split]

    train_x, test_x = X[train_indices, :], X[test_indices, :]

    if Constant.BRIX == property:
        train_y, test_y = y_brix[train_indices], y_brix[test_indices]
    else:
        # This experiment did not predict the hardness
        train_y, test_y = y_hardness[train_indices], y_hardness[test_indices]

    return train_x, train_y, test_x, test_y


def center_scale_y(Y):
    # center
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    # scale
    y_std = Y.std(axis=0, ddof=1)
    # y_std[y_std == 0.0] = 1.0
    Y /= y_std
    return Y, y_mean, y_std


def get_internal_self_dataloader(csv_path_list, batch_size, train_split_rate, property, test_batch_size, num_workers,
                                 y_is_standard):
    train_x = None
    train_y = None
    test_x = None
    test_y = None

    for i in range(len(csv_path_list)):
        csv_path = csv_path_list[i]
        item_train_x, item_train_y, item_test_x, item_test_y = split_train_test_data(csv_path,
                                                                                     train_split_rate,
                                                                                     property)
        if i == 0:
            train_x = item_train_x
            train_y = item_train_y
            test_x = item_test_x
            test_y = item_test_y
        else:
            train_x = np.vstack((train_x, item_train_x))
            train_y = np.hstack((train_y, item_train_y))
            test_x = np.vstack((test_x, item_test_x))
            test_y = np.hstack((test_y, item_test_y))

    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)

    plot_split_distribution(train_y, test_y)

    # Keep raw data without any processing
    train_raw_x = train_x.copy()
    test_raw_x = test_x.copy()
    train_raw_y = train_y.copy()
    test_raw_y = test_y.copy()
    raw_data = {
        'train_x': train_raw_x,
        'test_x': test_raw_x,
        'train_y': train_raw_y,
        'test_y': test_raw_y,
    }

    # Maximum absolute value scaling
    max_abs_scaler_x = MaxAbsScaler()
    train_x = max_abs_scaler_x.fit_transform(train_x)  # train_x
    test_x = max_abs_scaler_x.transform(test_x)  # test_x

    data_info = {}

    if y_is_standard:
        # Standardize y
        train_y, y_mean, y_std = center_scale_y(train_y)
        test_y -= y_mean
        test_y /= y_std
        # Save the mean and variance for later denormalization
        data_info['y_mean'] = y_mean
        data_info['y_std'] = y_std

    print('X range')
    print(np.min(train_x), np.max(train_x))
    print(np.min(test_x), np.max(test_x))

    print('Y range')
    print(np.min(train_y), np.max(train_y))
    print(np.min(test_y), np.max(test_y))

    train_data = SpectralDataset(X=train_x, y=train_y, mat=None, indexs=[1, 2, 3])
    test_data = SpectralDataset(X=test_x, y=test_y, mat=None, indexs=[1, 2, 3])

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, raw_data, data_info


def get_dataloaders(args):
    if args.dataset == Constant.blueberry:
        if args.test_mode == Constant.INTERNAL:
            # 80% for training and 20% for testing in each dataset
            # Put the mean and standard deviation used in standardization into data_info
            train_loader, test_loader, raw_data, data_info = get_internal_self_dataloader(args.train_csv_path_list,
                                                                                          args.batch_size,
                                                                                          args.train_split_rate,
                                                                                          args.property,
                                                                                          args.test_batch_size,
                                                                                          args.num_workers,
                                                                                          args.y_is_standard)

        else:
            data_info = {}
    else:
        pass

    data_info['y_is_standard'] = args.y_is_standard

    dls = {
        'train': train_loader,
        'test': test_loader,
        'raw_data': raw_data
    }
    return dls, data_info
