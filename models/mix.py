import numpy as np
import torch


def mixup(x, y, alpha=0.2):
    batch_size = x.shape[0]
    if alpha == 0.0:
        return x, y
    else:
        # Randomly sampling weights from beta distribution with parameter alpha
        weight = np.random.beta(alpha, alpha, batch_size)
        # weight = torch.Tensor(weight)
        weight = torch.Tensor(weight).cuda()

        x_weight = weight.reshape(batch_size, 1)
        y_weight = weight

        # Randomly sort numbers between [0, batch_size]
        index = np.random.permutation(batch_size)

        x1, x2 = x, x[index]
        x = x1 * x_weight + x2 * (1 - x_weight)

        y1, y2 = y, y[index]
        y = y1 * y_weight + y2 * (1 - y_weight)
        return x, y
