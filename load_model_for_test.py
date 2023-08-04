import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from dataloader import get_dataloaders
from models import model_loader
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import random

from config import Constant
from main import setup_seed

plt.rcParams['font.sans-serif'] = ['KaiTi']

"""
Load the trained model and predict the data
"""


def load_model_parse_args():
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, default=Constant.INTERNAL,
                        help='Test Mode of the code. "Constant.INTERNAL(internal)" ')

    parser.add_argument('--dataset', type=str, default=Constant.blueberry, help='')

    parser.add_argument('--train_csv_path_list', type=str, nargs='+',
                        default=Constant.train_csv_path_list,
                        help='The train csv data path list.')

    parser.add_argument('--train_mat_path_list', type=str, nargs='+',
                        default=Constant.train_mat_path_list,
                        help='The train mat data path list.')

    parser.add_argument('--train_split_rate', type=float, default=0.8, help='The train ratio used for the train split.')

    parser.add_argument('--refresh_rate', type=int, default=100, help='The refresh_rate of ProgressBar.')

    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the dataloader.')

    parser.add_argument('--model', type=str, default='MBA',
                        help='select use model [ablation_1|ablation_2|MBA')

    parser.add_argument('--y_is_standard', type=bool, default=True, help='y value whether standardization is required')

    # Model args
    parser.add_argument('--property', type=int, default=Constant.BRIX, help='the property will be predicted.')

    parser.add_argument('--checkpoint', type=str, default='', help='Model checkpoint path.')

    parser.add_argument('--mode', type=str, default='train', help='Mode of the code. Can be set to "train" or "test".')

    parser.add_argument('--epochs', type=int, default=5000, help='The number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size.')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='The test batch size.')
    parser.add_argument('--weight_decay', type=float, default=0.000, help='The parameter of L2 normalization.')

    # Parse
    args = parser.parse_args()
    return args


def test(args_temp, checkpoint_path):
    dls, data_info = get_dataloaders(args_temp)
    # Model
    model = model_loader.get_model(args_temp, data_info)
    # 加载模型
    model = model.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                       lr=0.001,
                                       weight_decay=0.000, loss=None,
                                       epochs=5000, data_info=[],
                                       batch_size=64)

    model.eval()
    test_dataloaders = dls['test']

    y_true_list = []
    y_pred_list = []

    y_mean = data_info['y_mean']
    y_std = data_info['y_std']

    for batch_index, data_item in enumerate(test_dataloaders):
        x = data_item[0]
        target = data_item[1]

        y_pred, w = model(x)
        y_pred = y_pred.reshape((1, -1))

        y_true_list.extend(target.tolist())
        y_pred_list.extend(y_pred[0].tolist())

    # Denormalization
    y_true_list = list(np.array(y_true_list) * y_std + y_mean)
    y_pred_list = list(np.array(y_pred_list) * y_std + y_mean)
    print(y_true_list)
    print(y_pred_list)

    # temp = [y_true_list, y_pred_list]
    # data2 = pd.DataFrame(data=temp, index=None)
    # data2.to_csv('./test_result_csv/' + args_temp.model + '_test_result.csv')


if __name__ == '__main__':
    setup_seed(1304)
    args = load_model_parse_args()

    args.model = 'MBA'
    checkpoint_path = 'checkpoint/1305/MBA_improve/5000_20230609130034/epoch=4999-step=65000.ckpt'

    test(args, checkpoint_path)
