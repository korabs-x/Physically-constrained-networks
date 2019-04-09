#!/usr/bin/env python3
#SBATCH --gpus=99
#SBATCH --mem=12GB
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abstreik

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

det_weights = [0, 1e-5, 1e-2, 0.1, 0.2, 0.5, 1, 2, 4]
# [:]

dim = 2
train_range = range(10, 21, 1)
# 21

SEED_TEST = 0


def get_data_loader(dim, n, seed=SEED_TEST, shuffle=False, batch_size=512):
    dataset = RotationDataset(dim, n, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def mp_worker(data):
    det_weight, n_train = data
    train_loader = get_data_loader(dim, n_train, seed=1683, shuffle=True, batch_size=512)
    test_loader = get_data_loader(dim, 512, seed=SEED_TEST, shuffle=False, batch_size=512)
    model = Net(dim, n_hidden_layers=max(1, int(math.log(dim, 2))))
    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'},
               {'loss_fn': lossfn.get_det_loss(), 'weight': det_weight, 'label': 'det'}]
    solver = Solver(model, loss_fn_train=loss_fn,
                    checkpoint_dir='checkpoints/checkpoints-detweight2/checkpoints_detweight-{}_dim-{}_ntrain-{}_seed-{}/'.format(
                        det_weight, dim, n_train, 1683))
    solver.train(train_loader, iterations=20000, test_every_iterations=200, test_loader=test_loader)
# 20000


def mp_handler():
    for det_weight in det_weights:
        for n_train in train_range:
            p = multiprocessing.Process(target=mp_worker, args=((det_weight, n_train),))
            p.start()


if __name__ == '__main__':
    mp_handler()
