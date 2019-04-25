#!/usr/bin/env python3
#SBATCH --gpus=120
#SBATCH --mem=12GB
#SBATCH --time=33:00:00
#SBATCH --mail-user=abstreik
#SBATCH --mail-type=ALL

import os
import multiprocessing
import lossfn
from experiment import run_experiment

loss_fns = ['error'] # ['error', 'det', 'norm', 'detnorm']

train_range = range(10, 21, 1)
n_runs = 20


def mp_worker(data):
    dim, lr, lossfnstr, train_seed = data
    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]
    if lossfnstr == 'det' or lossfnstr == 'detnorm':
        loss_fn.append({'loss_fn': lossfn.get_det_loss(), 'weight': 0.2, 'label': 'det'})
    if lossfnstr == 'norm' or lossfnstr == 'detnorm':
        loss_fn.append({'loss_fn': lossfn.get_norm_loss(), 'weight': 1, 'label': 'norm'})

    train_range_spec = train_range
    if dim == 2:
        train_range_spec = range(10, 21, 1)
    if dim == 3:
        train_range_spec = range(20, 201, 20)

    for n_train in train_range_spec:
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'checkpoints_lr_statistical/'
        checkpoint_dir += 'checkpoint_dim-{}_lr-{}_ntrain-{}_lossfn-{}_seed-{}/'.format(lr, dim, n_train, lossfnstr, train_seed)
        run_experiment(dim, n_train, train_seed, loss_fn, 0, checkpoint_dir, lr=lr)


def mp_handler():
    for dim in [2, 3]:
        for lossfnstr in loss_fns:
            for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]:
                for train_seed in range(1683, 1683 + n_runs, 1):
                    p = multiprocessing.Process(target=mp_worker, args=((dim, lr, lossfnstr, train_seed),))
                    p.start()


if __name__ == '__main__':
    mp_handler()
