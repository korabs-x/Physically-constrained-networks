#!/usr/bin/env python3
#SBATCH --gpus=200
#SBATCH --mem=12GB
#SBATCH --time=46:00:00
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

norm_weights = [0, 1e-2, 1e-1, 1e0, 1e1]

dims = [2, 3]
train_range = range(10, 21, 1)
n_runs = 20

SEED_TEST = 0


def mp_worker(data):
    norm_weight, dim, train_seed = data

    train_range_spec = train_range
    if dim == 2:
        train_range_spec = range(10, 21, 1)
    if dim == 3:
        train_range_spec = range(20, 201, 20)

    for n_train in train_range_spec:
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'round2_normweight/'
        checkpoint_dir += 'dim-{}_normweight-{}_ntrain-{}_seed-{}/'.format(dim, norm_weight, n_train, train_seed)
        loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'},
                   {'loss_fn': lossfn.get_norm_loss(), 'weight': norm_weight, 'label': 'norm'}]
        run_experiment(dim, n_train, train_seed, loss_fn, 0, checkpoint_dir, lr=1e-4, iterations=50000,
                       n_test=4096)


def mp_handler():
    for dim in dims:
        for norm_weight in norm_weights:
            for train_seed in range(1683, 1683 + n_runs, 1):
                p = multiprocessing.Process(target=mp_worker, args=((norm_weight, dim, train_seed),))
                p.start()


if __name__ == '__main__':
    mp_handler()
