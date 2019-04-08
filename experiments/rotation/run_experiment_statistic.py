#!/usr/bin/env python3
#SBATCH --gpus=80
#SBATCH --mem=12GB
#SBATCH --time=13:30:00
#SBATCH --mail-user=abstreik
#SBATCH --mail-type=ALL

import os
import multiprocessing
import lossfn
from experiment import run_experiment

loss_fns = ['error', 'det', 'norm', 'detnorm']# ['error', 'det', 'norm', 'detnorm']

dim = 2
train_range = range(7, 21, 1)
n_runs = 20

def mp_worker(data):
    lossfnstr, train_seed = data
    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]
    if lossfnstr == 'det' or lossfnstr == 'detnorm':
        loss_fn.append({'loss_fn': lossfn.get_det_loss(), 'weight': 0.2, 'label': 'det'})
    if lossfnstr == 'norm' or lossfnstr == 'detnorm':
        loss_fn.append({'loss_fn': lossfn.get_norm_loss(), 'weight': 0.2, 'label': 'norm'})

    for n_train in train_range:
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'checkpoints_statistical/'
        checkpoint_dir += 'checkpoint_dim-{}_ntrain-{}_lossfn-{}_seed-{}/'.format(dim, n_train, lossfnstr,
                                                                                      train_seed)
        run_experiment(dim, n_train, train_seed, loss_fn, 0, checkpoint_dir)


def mp_handler():
    for lossfnstr in loss_fns:
        for train_seed in range(1683, 1683 + n_runs, 1):
            p = multiprocessing.Process(target=mp_worker, args=((lossfnstr, train_seed),))
            p.start()


if __name__ == '__main__':
    mp_handler()
