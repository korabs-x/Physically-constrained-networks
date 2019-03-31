import torch.nn as nn
import torch


def get_mse_loss(**args_mse):
    loss_fn = nn.MSELoss(**args_mse)
    return lambda x, y, mat: loss_fn(x, y)


def get_det_loss():
    def loss_fn(x, y, mat):
        loss = 0
        for matrix in mat:
            loss += (torch.det(matrix) - 1) ** 2
        loss /= mat.shape[0]
        return loss
    return loss_fn


def get_norm_loss():
    def loss_fn(x, y, mat):
        loss = 0
        for pred in y:
            loss += (torch.norm(pred) - 1) ** 2
        loss /= y.shape[0]
        return loss
    return loss_fn


