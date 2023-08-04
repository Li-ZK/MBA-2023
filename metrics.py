import numpy as np
from sklearn import metrics

"""
Calculate various regression evaluation indicators based on true and predicted values
"""


def criterion_train(train_y, train_y_hat, is_standardization=False, y_mean=0.0, y_std=1.0):
    """
    Evaluate the training results
    :param train_y:
    :param train_y_hat:
    :param is_standardization:
    :param y_mean:
    :param y_std:
    :return:
    """

    # train_y = train_y.numpy()
    # train_y_hat = train_y_hat.numpy()

    # If the y value is standardized, then de standardization is required at this location
    if is_standardization:
        train_y = train_y * y_std + y_mean
        train_y_hat = train_y_hat * y_std + y_mean

    # train_y_hat = train_y_hat[:, 0]
    R2 = metrics.r2_score(y_true=train_y, y_pred=train_y_hat)
    MSE = metrics.mean_squared_error(train_y, train_y_hat)
    RMSEC = metrics.mean_squared_error(train_y, train_y_hat) ** 0.5
    MAE = metrics.mean_absolute_error(train_y, train_y_hat)
    return R2, MSE, RMSEC, MAE


def criterion_test(test_y, test_y_hat, is_standardization=False, y_mean=0.0, y_std=1.0):
    """
    Evaluate the test set
    :param test_y:
    :param test_y_hat:
    :param is_standardization:
    :param y_mean:
    :param y_std:
    :return:
    """

    # test_y = test_y.numpy()
    # test_y_hat = test_y_hat.numpy()

    if is_standardization:
        # Denormalization
        test_y = test_y * y_std + y_mean
        test_y_hat = test_y_hat * y_std + y_mean

    # test_y_hat = test_y_hat[:, 0]

    R2 = metrics.r2_score(y_pred=test_y_hat, y_true=test_y)
    MSE = metrics.mean_squared_error(test_y, test_y_hat)
    RMSEP = metrics.mean_squared_error(test_y, test_y_hat) ** 0.5
    MAE = metrics.mean_absolute_error(test_y, test_y_hat)
    SDR = test_y.std() / RMSEP
    # PG = train_y.std() / RMSEP
    return R2, MSE, RMSEP, MAE, SDR
