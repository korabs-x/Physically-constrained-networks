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


def get_norm_loss_old():
    def loss_fn(pred, y, mat):
        loss = 0
        for pred_row in pred:
            loss += (torch.norm(pred_row) - 1) ** 2
        loss /= y.shape[0]
        return loss
    return loss_fn


def get_norm_loss(**args_mse):
    # calculates the mse loss between the prediction and the normed prediction
    mse = nn.MSELoss(**args_mse)
    # mse_no_reduction = nn.MSELoss(reduction='none')

    def loss_fn(pred, y, mat):
        norms = torch.norm(pred, 2, 1)
        # sqnorms = torch.norm(pred, 2, 1) ** 2
        # sqnorms = torch.sum(mse_no_reduction(pred, torch.zeros_like(pred)), dim=1)
        # return mse(sqnorms, torch.ones_like(sqnorms))
        return mse(pred, pred / norms.view(norms.shape[0], 1))

    return loss_fn
