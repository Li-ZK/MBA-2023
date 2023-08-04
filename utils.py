import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar

import os
import numpy as np
import random
from matplotlib import pyplot as plt

import datetime


def get_now_time_yyyymmddhhMMss():
    now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return now_time


def get_callbacks(args):
    callbacks = []

    # Model checkpoint
    # model_checkpoint_clbk = pl.callbacks.model_checkpoint.ModelCheckpoint(
    #     dirpath='checkpoint/' + str(args.seed) + '/' + args.model + '/' + str(
    #         args.epochs) + '_' + get_now_time_yyyymmddhhMMss() + '/',
    #     save_last=True,
    # )
    #
    # model_checkpoint_clbk.CHECKPOINT_NAME_LAST = args.model + '+' + str(
    #     random.randint(1, 100)) + '+{epoch}-{step}-{test loss:.2f}'
    #
    # callbacks += [model_checkpoint_clbk]
    callbacks += [TQDMProgressBar(refresh_rate=args.refresh_rate)]
    return callbacks


def get_logger(args):
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), 'logger'),
        name=args.model,
    )
    tb_logger.log_hyperparams(args)
    return tb_logger


class MetricsRecordUtil:
    """
    Tools for recording various measurement indicators
    R2,MSE,RMSEC,MAE,...
    """

    def __init__(self):
        self.train_loss_list = []
        self.train_R2_list = []
        self.train_MSE_list = []
        self.train_RMSEC_list = []
        self.train_MAE_list = []

        self.test_loss_list = []
        self.test_R2_list = []
        self.test_MSE_list = []
        self.test_RMSEP_list = []
        self.test_MAE_list = []

        self.SDR_list = []

    def add_train_metrics(self, loss, R2, MSE, RMSEC, MAE):
        self.train_loss_list.append(loss)
        self.train_R2_list.append(R2)
        self.train_MSE_list.append(MSE)
        self.train_RMSEC_list.append(RMSEC)
        self.train_MAE_list.append(MAE)

    def add_test_metrics(self, loss, R2, MSE, RMSEP, MAE, SDR):
        self.test_loss_list.append(loss)
        self.test_R2_list.append(R2)
        self.test_MSE_list.append(MSE)
        self.test_RMSEP_list.append(RMSEP)
        self.test_MAE_list.append(MAE)
        self.SDR_list.append(SDR)

    def print_train_result(self):
        """
        Select the indicators of the last epoch as the final result
        :return:
        """
        result = {}
        best_result = {}
        # train
        train_res_loss = self.train_loss_list[-1]
        train_res_r2 = self.train_R2_list[-1]
        train_res_rmse = self.train_RMSEC_list[-1]
        train_res_mae = self.train_MAE_list[-1]
        # test
        test_res_loss = self.test_loss_list[-1]
        test_res_r2 = self.test_R2_list[-1]
        test_res_rmse = self.test_RMSEP_list[-1]
        test_res_mae = self.test_MAE_list[-1]
        res_sdr = self.SDR_list[-1]

        # best result(Just recording, the final experimental results will be based on the last epoch result)
        max_index = np.argmax(self.test_R2_list)
        res_info = 'epoch of best result: ' + str(max_index + 1) + "\n"
        res_info += "train loss: {:.4f}, train R2: {:.4f}, train RMSEC: {:.4f}, train MSE: {:.4f},train MAE: {:.4f}".format(
            self.train_loss_list[max_index],
            self.train_R2_list[max_index],
            self.train_RMSEC_list[max_index],
            self.train_RMSEC_list[max_index] * self.train_RMSEC_list[max_index],
            self.train_MAE_list[max_index])
        res_info += "\n"

        res_info += "test loss: {:.4f}, test R2: {:.4f}, test RMSEP: {:.4f},test MSE: {:.4f}, test MAE: {:.4f}".format(
            self.test_loss_list[max_index],
            self.test_R2_list[max_index],
            self.test_RMSEP_list[max_index],
            self.test_RMSEP_list[max_index] * self.test_RMSEP_list[max_index],
            self.test_MAE_list[max_index],
        )
        res_info += "\n" + "SDR: {:.4f}".format(self.SDR_list[max_index])
        # save best result to dict
        best_result['train loss'] = format(self.train_loss_list[max_index], '.4f')
        best_result['train R2'] = format(self.train_R2_list[max_index], '.4f')
        best_result['train RMSEC'] = format(self.train_RMSEC_list[max_index], '.4f')
        best_result['train MSE'] = format(self.train_RMSEC_list[max_index] * self.train_RMSEC_list[max_index], '.4f')
        best_result['train MAE'] = format(self.train_MAE_list[max_index], '.4f')

        best_result['test loss'] = format(self.test_loss_list[max_index], '.4f')
        best_result['test R2'] = format(self.test_R2_list[max_index], '.4f')
        best_result['test RMSEP'] = format(self.test_RMSEP_list[max_index], '.4f')
        best_result['test MSE'] = format(self.test_RMSEP_list[max_index] * self.test_RMSEP_list[max_index], '.4f')
        best_result['test MAE'] = format(self.test_MAE_list[max_index], '.4f')
        best_result['SDR'] = format(self.SDR_list[max_index], '.4f')

        # final result
        res_info += "\n" + 'epoch of final result:' + "\n"
        res_info += "train loss: {:.4f}, train R2: {:.4f}, train RMSEC: {:.4f},train MSE: {:.4f}, train MAE: {:.4f}".format(
            train_res_loss,
            train_res_r2,
            train_res_rmse,
            train_res_rmse * train_res_rmse,
            train_res_mae)
        res_info += "\n" + "test loss: {:.4f}, test R2: {:.4f}, test RMSEP: {:.4f},test MSE: {:.4f}, test MAE: {:.4f}".format(
            test_res_loss,
            test_res_r2,
            test_res_rmse,
            test_res_rmse * test_res_rmse,
            test_res_mae)
        res_info += "\n" + "SDR: {:.4f}".format(res_sdr)
        # save final result to dict
        result['train loss'] = format(train_res_loss, '.4f')
        result['train R2'] = format(train_res_r2, '.4f')
        result['train RMSEC'] = format(train_res_rmse, '.4f')
        result['train MSE'] = format(train_res_rmse * train_res_rmse, '.4f')
        result['train MAE'] = format(train_res_mae, '.4f')

        result['test loss'] = format(test_res_loss, '.4f')
        result['test R2'] = format(test_res_r2, '.4f')
        result['test RMSEP'] = format(test_res_rmse, '.4f')
        result['test MSE'] = format(test_res_rmse * test_res_rmse, '.4f')
        result['test MAE'] = format(test_res_mae, '.4f')
        result['SDR'] = format(res_sdr, '.4f')

        print(res_info)

        # save info to file
        return result, best_result, res_info

    def plot(self, flag, interval, smoothing_factor=0.9):

        def smooth_curve(points, factor=0.9):
            smoothed_points = []
            for point in points:
                if smoothed_points:
                    previous = smoothed_points[-1]
                    smoothed_points.append(previous * factor + point * (1 - factor))
                else:
                    smoothed_points.append(point)
            return smoothed_points

        train_losses = smooth_curve(self.train_loss_list, smoothing_factor)
        test_losses = smooth_curve(self.test_loss_list, smoothing_factor)

        num_epoch = len(train_losses)

        flag = flag // interval
        plt.figure(figsize=(8, 4.5))
        x_axis_value = [i for i in range(0, num_epoch, interval)]
        train_value = []
        test_value = []
        for i in range(len(x_axis_value)):
            train_value.append(train_losses[x_axis_value[i]])
            test_value.append(test_losses[x_axis_value[i]])

        plt.rcParams['axes.unicode_minus'] = False

        with plt.style.context(('ggplot')):
            plt.plot(x_axis_value[flag:], train_value[flag:], alpha=0.8, linewidth=1, linestyle='solid',
                     label='train loss')

            plt.plot(x_axis_value[flag:], test_value[flag:], alpha=0.8, linewidth=1, linestyle='solid',
                     label='test loss')

            plt.title("loss" + str(flag))
            plt.legend()
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.show()

    def save(self):
        # save every metrics list to file
        pass
