#!/usr/bin/env python3
# SBATCH --cpus=80
# SBATCH --mem=12GB
# SBATCH --time=24:00:00
# SBATCH --mail-user=abstreik
# SBATCH --mail-type=ALL

import os
import multiprocessing
import lossfn
from experiment import run_experiment

loss_fns = ['error', 'det', 'norm', 'detnorm']

dim = 2
train_range = range(1, 3, 1) # 31
n_runs = 2 # 20

def mp_worker(data):
    lossfn, train_seed = data
    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]
    if lossfn == 'det' or lossfn == 'detnorm':
        loss_fn.append({'loss_fn': lossfn.get_det_loss(), 'weight': 0.5, 'label': 'det'})
    if lossfn == 'norm' or lossfn == 'detnorm':
        loss_fn.append({'loss_fn': lossfn.get_norm_loss(), 'weight': 0.5, 'label': 'norm'})

    for n_train in train_range:
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'checkpoints_statistical/'
        checkpoint_dir += 'checkpoint_dim-{}_ntrain-{}_lossfn-{}_seed-{}'.format(dim, n_train, lossfn,
                                                                                      train_seed)
        run_experiment(dim, n_train, train_seed, loss_fn, weight_decay, checkpoint_dir)


def mp_handler():
    for lossfn in loss_fns:
        for train_seed in range(1683, 1683 + n_runs, 1):
            p = multiprocessing.Process(target=mp_worker, args=((lossfn, train_seed),))
            p.start()


if __name__ == '__main__':
    mp_handler()
