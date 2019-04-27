#!/usr/bin/env python3
#SBATCH --gpus=40
#SBATCH --mem=12GB
#SBATCH --time=120:00:00
#SBATCH --mail-user=abstreik
#SBATCH --mail-type=ALL

import os
import multiprocessing
import lossfn
import torch
import numpy as np
import math
import argparse
from experiment import run_experiment

dims = [2, 3]

SEED_TEST = 0
# train_seed = 1683

n_runs = 20


def get_fn_det_normed(dim):
    def fn_det_normed(mats):
        scale_tensor = torch.ones(size=(mats.shape[0],))
        for i, mat in enumerate(mats):
            det = torch.det(mat.detach())
            if det > 0:
                scale_tensor[i] = det**(1./dim)
        scale_tensor = scale_tensor.view(scale_tensor.shape[0], -1)
        return (mats.view(mats.shape[0], -1) / scale_tensor).view(mats.shape[0], dim, dim)
    return fn_det_normed


def fn_det_unnormed(mats):
    return mats


def mp_worker(data):
    dim, fn_matrix_name, train_seed = data

    fn_matrix = get_fn_det_normed(dim) if fn_matrix_name == "normed" else fn_det_unnormed

    train_range = range(10, 21, 1) if dim == 2 else range(20, 201, 20)
    for n_train in train_range:
        loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'round2_normed_det/'
        checkpoint_dir += 'dim-{}_normeddet-{}_ntrain-{}_seed-{}/'.format(
            fn_pred_name, dim, n_train, train_seed)
        run_experiment(dim, n_train, train_seed, loss_fn, 0, checkpoint_dir, fn_matrix=fn_matrix, lr=5e-5, iterations=50000, n_test=4096)


def mp_handler():
    for dim in dims:
        for fn_det_name in ["normed", "orig"]:
            for train_seed in range(1683, 1683 + n_runs, 1):
                p = multiprocessing.Process(target=mp_worker, args=((dim, fn_det_name, train_seed),))
                p.start()


if __name__ == '__main__':
    mp_handler()
