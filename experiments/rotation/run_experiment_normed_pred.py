#!/usr/bin/env python3
#SBATCH --gpus=4
#SBATCH --mem=12GB
#SBATCH --time=120:00:00
#SBATCH --mail-user=abstreik
#SBATCH --mail-type=ALL

import os
import multiprocessing
from solver import Solver
from model import Net
from dataset import RotationDataset
import lossfn
from torch.utils.data import DataLoader
import torch
import numpy as np
import math
import argparse
from experiment import run_experiment

dims = [2, 3]

SEED_TEST = 0
train_seed = 1683


def fn_pred_normed(pred):
    norms = torch.norm(pred.data, 2, 1)
    pred_normed = pred / norms.view(norms.shape[0], 1)
    return pred_normed


def fn_pred_unnormed(pred):
    return pred


def get_data_loader(dim, n, seed=SEED_TEST, shuffle=False, batch_size=512):
    dataset = RotationDataset(dim, n, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def mp_worker(data):
    dim, fn_pred_name = data

    fn_pred = fn_pred_normed if fn_pred_name == "normed" else fn_pred_unnormed

    train_range = range(1, 21, 1) if dim == 2 else range(1, 41, 2)
    for n_train in train_range:
        loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]
        checkpoint_dir = 'checkpoints/checkpoints-normedpred/checkpoints_normedpred-{}_dim-{}_ntrain-{}_seed-{}/'.format(
            fn_pred_name, dim, n_train, train_seed)
        run_experiment(dim, n_train, train_seed, loss_fn, 0, checkpoint_dir, fn_pred=fn_pred, iterations=50000)


def mp_handler():
    for dim in dims:
        for fn_pred_name in ["normed", "orig"]:
            p = multiprocessing.Process(target=mp_worker, args=((dim, fn_pred_name),))
            p.start()


if __name__ == '__main__':
    mp_handler()
