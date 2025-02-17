#!/usr/bin/env python3
#SBATCH --gpus=360
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

det_weights = [0, 1e-5, 1e-2, 0.1, 0.2, 0.5, 1, 2, 4]

train_range = range(10, 21, 1)

SEED_TEST = 0
n_runs = 20


def mp_worker(data):
    dim, det_weight, n_train = data

    for train_seed in range(1683, 1683 + n_runs, 1):
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'round2_detweight4/'
        checkpoint_dir += 'dim-{}_detweight-{}_ntrain-{}_seed-{}/'.format(dim, det_weight, n_train, train_seed)
        loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'},
                   {'loss_fn': lossfn.get_det_loss(), 'weight': det_weight, 'label': 'det'}]
        run_experiment(dim, n_train, train_seed, loss_fn, 0, checkpoint_dir, lr=5e-5, iterations=50000,
                       n_test=4096)


def mp_handler():
    for dim in [2]:
        for det_weight in det_weights:
            train_range_spec = train_range
            if dim == 2:
                train_range_spec = range(10, 21, 1)
            if dim == 3:
                train_range_spec = range(20, 201, 20)

            for n_train in train_range_spec:
                p = multiprocessing.Process(target=mp_worker, args=((dim, det_weight, n_train),))
                p.start()


if __name__ == '__main__':
    mp_handler()
