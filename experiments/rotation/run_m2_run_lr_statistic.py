#!/usr/bin/env python3
#SBATCH --gpus=60
#SBATCH --mem=12GB
#SBATCH --time=46:00:00
#SBATCH --mail-user=abstreik
#SBATCH --mail-type=ALL

import os
import multiprocessing
import lossfn
from experiment import run_experiment
from model import NetConnected100

loss_fns = ['error']

train_range = range(10, 21, 1)
n_runs = 20


def mp_worker(data):
    dim, lr, lossfnstr, n_train = data
    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]

    for train_seed in range(1683, 1683 + n_runs, 1):
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'm2_lr_statistical/'
        checkpoint_dir += 'dim-{}_lr-{}_ntrain-{}_lossfn-{}_seed-{}/'.format(dim, lr, n_train, lossfnstr, train_seed)
        run_experiment(dim, n_train, train_seed, loss_fn, 0, checkpoint_dir, lr=lr, iterations=50000, n_test=4096,
                       model_class=NetConnected100)


def mp_handler():
    for dim in [2]:
        for lossfnstr in loss_fns:
            for lr in [1e-5, 1e-4, 1e-3, 1e-2]:
                train_range_spec = train_range
                if dim == 2:
                    train_range_spec = range(10, 21, 1)
                if dim == 3:
                    train_range_spec = range(20, 201, 20)
                for n_train in train_range_spec:
                    p = multiprocessing.Process(target=mp_worker, args=((dim, lr, lossfnstr, n_train),))
                    p.start()


if __name__ == '__main__':
    mp_handler()
