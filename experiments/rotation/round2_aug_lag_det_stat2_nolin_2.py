#!/usr/bin/env python3
#SBATCH --gpus=11
#SBATCH --mem=12GB
#SBATCH --time=120:00:00
#SBATCH --mail-user=abstreik
#SBATCH --mail-type=ALL

import os
import multiprocessing
import lossfn
from experiment import run_experiment_augmented_lagrangian

loss_fns = ['error']

train_range = range(10, 21, 1)
n_runs = 20
dims = [2]

config = {'weights': [5*1e-3] * 30, 'iterations': [5000] * 30}
# config = {'weights': [1e-2] * 15 + [1e-1] * 15, 'iterations': [5000] * 30},


def mp_worker(data):
    dim, n_train = data

    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]
    lin_constraints = [{'fn': lossfn.det_linear, 'label': 'det'}]

    for train_seed in range(1683, 1683 + n_runs, 1):
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'round2_aug_lag_det_fixed_nolin/'
        checkpoint_dir += 'config_1/'
        checkpoint_dir += 'dim-{}_ntrain-{}_seed-{}/'.format(dim, n_train, train_seed)

        run_experiment_augmented_lagrangian(dim, n_train, train_seed, loss_fn, lin_constraints, config['weights'],
                                            checkpoint_dir, iterations=config['iterations'], exclude_linear=True)


def mp_handler():
    for dim in dims:
        for n_train in train_range:
            p = multiprocessing.Process(target=mp_worker, args=((dim, n_train),))
            p.start()


if __name__ == '__main__':
    mp_handler()
