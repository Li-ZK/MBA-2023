import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from dataloader import get_dataloaders
from models import model_loader
import argparse
import matplotlib.pyplot as plt
import random

from config import Constant
from main import setup_seed

plt.rcParams['font.sans-serif'] = ['KaiTi']

"""
Load the trained MBA model, calculate band weights, and draw a graph
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


wavelength = [393.700000, 396.900000, 400.100000, 403.200000, 406.400000, 409.600000,
              412.800000, 416.000000, 419.200000, 422.400000, 425.600000, 428.800000,
              432.000000, 435.300000, 438.500000, 441.700000, 444.900000, 448.200000,
              451.400000, 454.600000, 457.900000, 461.100000, 464.400000, 467.600000,
              470.900000, 474.100000, 477.400000, 480.700000, 483.900000, 487.200000,
              490.500000, 493.700000, 497.000000, 500.300000, 503.600000, 506.900000,
              510.200000, 513.500000, 516.800000, 520.100000, 523.400000, 526.700000,
              530.000000, 533.400000, 536.700000, 540.000000, 543.300000, 546.700000,
              550.000000, 553.300000, 556.700000, 560.000000, 563.400000, 566.700000,
              570.100000, 573.400000, 576.800000, 580.200000, 583.600000, 586.900000,
              590.300000, 593.700000, 597.100000, 600.500000, 603.800000, 607.200000,
              610.600000, 614.000000, 617.400000, 620.900000, 624.300000, 627.700000,
              631.100000, 634.500000, 637.900000, 641.400000, 644.800000, 648.200000,
              651.700000, 655.100000, 658.600000, 662.000000, 665.500000, 668.900000,
              672.400000, 675.800000, 679.300000, 682.800000, 686.200000, 689.700000,
              693.200000, 696.700000, 700.200000, 703.700000, 707.200000, 710.700000,
              714.200000, 717.700000, 721.200000, 724.700000, 728.200000, 731.700000,
              735.200000, 738.700000, 742.300000, 745.800000, 749.300000, 752.900000,
              756.400000, 760.000000, 763.500000, 767.100000, 770.600000, 774.200000,
              777.700000, 781.300000, 784.900000, 788.400000, 792.000000, 795.600000,
              799.200000, 802.800000, 806.300000, 809.900000, 813.500000, 817.100000,
              820.700000, 824.300000, 827.900000, 831.600000, 835.200000, 838.800000,
              842.400000, 846.000000, 849.700000, 853.300000, 856.900000, 860.600000,
              864.200000, 867.900000, 871.500000, 875.200000, 878.800000, 882.500000,
              886.200000, 889.800000, 893.500000, 897.200000, 900.800000, 904.500000,
              908.200000, 911.900000, 915.600000, 919.300000, 923.000000, 926.700000,
              930.400000, 934.100000, 937.800000, 941.500000, 945.200000, 949.000000,
              952.700000, 956.400000, 960.100000, 963.900000, 967.600000, 971.300000,
              975.100000, 978.800000, 982.600000, 986.400000, 990.100000, 993.900000,
              997.600000, 1001.400000]


def plot_feature_wavelength_curve(wavelength, X, val_sel_indexs):
    plt.figure(figsize=(32, 15))
    fontsize = 50
    plt.grid(False)

    for index in val_sel_indexs:
        # plt.scatter(wavelength[index], 0, s=60, marker='^', color='r')
        plt.axvline(wavelength[index], c="r", ls="--", lw=2)

    with plt.style.context(('ggplot')):
        plt.plot(wavelength, X.T)
        plt.annotate('Top 10 bands', xy=(935, 0.2), fontsize=38)
        plt.xlabel('Wavelength(nm)', fontsize=fontsize)
        plt.ylabel('Reflectance', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

    # plt.savefig('./feature_image/' + save_name + '_size.pdf', dpi=500)
    plt.show()


def test(args_temp):
    setup_seed(1305)
    checkpoint_path = 'checkpoint/1305/MBA_improve/5000_20230609130034/epoch=4999-step=65000.ckpt'

    dls, data_info = get_dataloaders(args_temp)

    raw_train_x = dls['raw_data']['train_x']

    # plot_curve(wavelength, raw_train_x[:100, :])

    # Model
    model = model_loader.get_model(args_temp, data_info)
    model = model.load_from_checkpoint(checkpoint_path=checkpoint_path, lr=0.001,
                                       weight_decay=0.000, loss=None,
                                       epochs=5000, data_info=[], batch_size=64)

    model.eval()
    loss = nn.MSELoss(reduction='mean')
    train_dataloaders = dls['train']

    train_loss = 0
    band_weight = None
    for batch_index, data_item in enumerate(train_dataloaders):
        x = data_item[0]
        target = data_item[1]
        target = torch.reshape(target, (x.size()[0], 1))

        y_pred, w = model(x)
        temp_loss = loss(y_pred, target)

        train_loss += temp_loss

        w = torch.sum(w, dim=0)
        if band_weight is None:
            band_weight = w
        else:
            band_weight = band_weight + w

    print('train loss', train_loss)
    k = 10
    # Obtain weights for all bands in the training sample, sorted from largest to smallest
    band_weights = band_weight[0].sort(0, True)

    weights = band_weights[0]
    print('weight', weights[:k])

    index = band_weights[1].tolist()  # Corresponding index after sorting
    print(index)
    bands_nm_list = []
    for i in range(k):
        bands_nm_list.append(str(wavelength[index[i]]) + 'nm')

    feature_wavelenght_index = index[:k]
    print('bands', bands_nm_list)
    print('indexs', feature_wavelenght_index)

    plot_feature_wavelength_curve(wavelength, raw_train_x[:100], feature_wavelenght_index)


if __name__ == '__main__':
    args = load_model_parse_args()
    test(args)
