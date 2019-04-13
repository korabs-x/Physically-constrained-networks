#!/usr/bin/env python3
#SBATCH --gpus=8
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

norm_weights = [0, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
# [:]

dims = [3]
train_range = range(5, 101, 5)
# 21

SEED_TEST = 0


def get_data_loader(dim, n, seed=SEED_TEST, shuffle=False, batch_size=512):
    dataset = RotationDataset(dim, n, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def mp_worker(data):
    norm_weight, dim = data
    for n_train in train_range:
        train_loader = get_data_loader(dim, n_train, seed=1683, shuffle=True, batch_size=512)
        test_loader = get_data_loader(dim, 512, seed=SEED_TEST, shuffle=False, batch_size=512)
        model = Net(dim, n_hidden_layers=max(1, int(math.log(dim, 2))))
        loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'},
                   {'loss_fn': lossfn.get_norm_loss(), 'weight': norm_weight, 'label': 'norm'}]
        solver = Solver(model, loss_fn_train=loss_fn,
                        checkpoint_dir='checkpoints/checkpoints-normweight_new/checkpoints_normweight-{}_dim-{}_ntrain-{}_seed-{}/'.format(
                            norm_weight, dim, n_train, 1683))
        solver.train(train_loader, iterations=20000, test_every_iterations=200, test_loader=test_loader)


def mp_handler():
    for norm_weight in norm_weights:
        for dim in dims:
            p = multiprocessing.Process(target=mp_worker, args=((norm_weight, dim),))
            p.start()


if __name__ == '__main__':
    mp_handler()
