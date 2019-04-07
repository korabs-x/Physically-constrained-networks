import torch.nn as nn
import torch


def get_mse_loss(**args_mse):
    loss_fn = nn.MSELoss(**args_mse)
    return lambda pred, y, mat: loss_fn(pred, y)


def get_det_loss():
    def loss_fn(pred, y, mat):
        loss = 0
        for matrix in mat:
            loss += (torch.det(matrix) - 1) ** 2
        loss /= mat.shape[0]
        return loss
    return loss_fn


def get_norm_loss():
    def loss_fn(pred, y, mat):
        loss = 0
        for pred_row in pred:
            loss += (torch.norm(pred_row) - 1) ** 2
        loss /= y.shape[0]
        return loss
    return loss_fn


