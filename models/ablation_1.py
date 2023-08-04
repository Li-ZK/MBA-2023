import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl

from metrics import criterion_train, criterion_test
from utils import MetricsRecordUtil
from schedulers import WarmupCosineLR

"""
ablation_1
Regression Network
"""


class BasicConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        """

        :param in_channels:
        :param out_channels:
        :param kwargs: kernel_size,stride,padding,etc.
        """
        super(BasicConv1d, self).__init__()
        self.Conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.BN = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.Conv(x)
        out = self.BN(out)
        return out


class InceptionResNetA(nn.Module):
    def __init__(self, in_channels, mid_channels):
        """
        default: stride=1,padding=0
        :param in_channels:
        """
        super(InceptionResNetA, self).__init__()
        self.b1 = BasicConv1d(in_channels, mid_channels, kernel_size=1)

        self.b2_1 = BasicConv1d(in_channels, mid_channels, kernel_size=1)
        self.b2_2 = BasicConv1d(mid_channels, mid_channels, kernel_size=3, padding=1)

        self.b3_1 = BasicConv1d(in_channels, mid_channels, kernel_size=1)
        self.b3_2 = BasicConv1d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.b3_3 = BasicConv1d(mid_channels, mid_channels, kernel_size=3, padding=1)

        self.tb = BasicConv1d(3 * mid_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = F.leaky_relu(x)

        b_out1 = F.leaky_relu(self.b1(x))
        b_out2 = F.leaky_relu(self.b2_2(F.leaky_relu(self.b2_1(x))))
        b_out3 = F.leaky_relu(self.b3_3(F.leaky_relu(self.b3_2(F.leaky_relu(self.b3_1(x))))))
        b_out = torch.cat([b_out1, b_out2, b_out3], 1)

        b_out = self.tb(b_out)

        y = b_out + x

        out = F.leaky_relu(y)
        return out


class InceptionResNet(nn.Module):
    """
    Feature Extractor
    """

    def __init__(self):
        super(InceptionResNet, self).__init__()

        self.stem = nn.Sequential(
            nn.BatchNorm1d(1),
            BasicConv1d(1, 32, kernel_size=7, stride=3),
        )

        self.irA = InceptionResNetA(in_channels=32, mid_channels=64)

        self.FNN = nn.Sequential(
            nn.Flatten(),

            nn.Linear(1824, 128),

            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        out = self.stem(x)
        out = self.irA(out)
        out = self.FNN(out)
        return out


class ablation_1(nn.Module):

    def __init__(self):
        super(ablation_1, self).__init__()
        self.regression_network = InceptionResNet()

    def forward(self, x):
        y_pred = self.regression_network(x)
        return y_pred


class LightningModel(pl.LightningModule):

    def __init__(self, lr, weight_decay, loss, epochs, data_info, batch_size):
        super(LightningModel, self).__init__()
        # define model
        self.Net = ablation_1()

        # define model parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.data_info = data_info
        self.record = MetricsRecordUtil()

        # define model loss
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, x):
        # forward defines the forward
        y_pred = self.Net(x)
        return y_pred

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop(one batch).
        # receive and process data
        x = batch[0]
        target = batch[1]
        mat = batch[2]

        target = torch.reshape(target, (x.size()[0], 1))

        # forward,get pred
        y_pred = self(x)  # y_pred

        # calculate loss
        loss = self.loss(y_pred, target)

        # calculate metrics
        R2, MSE, RMSEC, MAE = criterion_train(train_y=target.cpu().detach().numpy(),
                                              train_y_hat=y_pred.cpu().detach().numpy(),
                                              is_standardization=self.data_info['y_is_standard'],
                                              y_mean=self.data_info['y_mean'],
                                              y_std=self.data_info['y_std'])

        training_outputs = {
            "loss": loss,
            "R2": R2,
            "MSE": MSE,
            "RMSEC": RMSEC,
            "MAE": MAE,
        }

        return training_outputs

    def training_epoch_end(self, training_outputs):
        avg_loss = torch.tensor([x["loss"] for x in training_outputs]).mean()
        avg_R2 = torch.tensor([x["R2"] for x in training_outputs]).mean()
        avg_MSE = torch.tensor([x["MSE"] for x in training_outputs]).mean()
        avg_RMSEC = torch.tensor([x["RMSEC"] for x in training_outputs]).mean()
        avg_MAE = torch.tensor([x["MAE"] for x in training_outputs]).mean()

        # log
        self.log('train MSE', avg_MSE, logger=False)
        self.log('train RMSEC', avg_RMSEC, logger=False)
        self.log('train MAE', avg_MAE, logger=False)
        self.log('train R2', avg_R2, prog_bar=True)
        self.log('train loss', avg_loss, prog_bar=True)

        # save metrics to record
        self.record.add_train_metrics(loss=avg_loss, R2=avg_R2, MSE=avg_MSE, RMSEC=avg_RMSEC, MAE=avg_MAE)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        # receive and process data
        x = batch[0]
        target = batch[1]
        mat = batch[2]

        target = torch.reshape(target, (x.size()[0], 1))

        # forward,get pred
        y_pred = self(x)  # y_pred

        # calculate loss
        loss = self.loss(y_pred, target)

        # calculate metrics
        R2, MSE, RMSEP, MAE, SDR = criterion_test(test_y=target.cpu().detach().numpy(),
                                                  test_y_hat=y_pred.cpu().detach().numpy(),
                                                  is_standardization=self.data_info['y_is_standard'],
                                                  y_mean=self.data_info['y_mean'],
                                                  y_std=self.data_info['y_std'])

        validation_outputs = {
            "loss": loss,
            "R2": R2,
            "MSE": MSE,
            "RMSEP": RMSEP,
            "MAE": MAE,
            "SDR": SDR,
        }

        return validation_outputs

    def validation_epoch_end(self, validation_outputs):
        avg_loss = torch.tensor([x["loss"] for x in validation_outputs]).mean()
        avg_R2 = torch.tensor([x["R2"] for x in validation_outputs]).mean()
        avg_MSE = torch.tensor([x["MSE"] for x in validation_outputs]).mean()
        avg_RMSEP = torch.tensor([x["RMSEP"] for x in validation_outputs]).mean()
        avg_MAE = torch.tensor([x["MAE"] for x in validation_outputs]).mean()
        avg_SDR = torch.tensor([x["SDR"] for x in validation_outputs]).mean()

        # log
        self.log('test MSE', avg_MSE, logger=False)
        self.log('test MAE', avg_MAE, logger=False)
        self.log('test RMSEP', avg_RMSEP, logger=False)
        self.log('test SDR', avg_SDR, logger=False)
        self.log('test R2', avg_R2, prog_bar=True)
        self.log('test loss', avg_loss, prog_bar=True)

        # save metrics to record
        self.record.add_test_metrics(loss=avg_loss, R2=avg_R2, MSE=avg_MSE, RMSEP=avg_RMSEP, MAE=avg_MAE, SDR=avg_SDR)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        print('test step')
        x = batch[0]
        target = batch[1]
        mat = batch[2]

        target = torch.reshape(target, (x.size()[0], 1))

        # forward,get pred
        x_rec, y_pred = self(x)  # y_pred

        y_pred = y_pred.cpu().detach().numpy()

        test_outputs = {
            "y_pred": y_pred,
        }

        return test_outputs

    def test_epoch_end(self, test_outputs):
        print('test epoch end')
        return

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)

        T_max = self.epochs
        eta_min = 1e-9

        scheduler = WarmupCosineLR(optimizer, lr_min=eta_min, lr_max=self.lr, warm_up=100, T_max=T_max)

        self.logger.log_hyperparams({'T_max': T_max, 'eta_min': eta_min})

        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict


def create_model(args, data_info):
    loss = nn.MSELoss(reduction='mean')

    # extract model args
    model_args = {
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'loss': loss,
        'epochs': args.epochs,
        'data_info': data_info,
        'batch_size': args.batch_size,
    }
    model = LightningModel(**model_args)

    # init model weight
    # init_weights(model, args.init_type)

    def init_xavier(m):
        if type(m) == nn.Conv1d:
            nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    model.apply(init_xavier)

    if len(args.checkpoint) > 0:
        model = model.load_from_checkpoint(args.checkpoint, **model_args)
        print(f'Loaded model checkpoint {args.checkpoint}')
    return model
