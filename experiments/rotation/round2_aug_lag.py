#!/usr/bin/env python3
#SBATCH --gpus=15
#SBATCH --mem=12GB
#SBATCH --time=46:00:00
#SBATCH --mail-user=abstreik
#SBATCH --mail-type=ALL

import os
import multiprocessing
import lossfn
from experiment import run_experiment_augmented_lagrangian

loss_fns = ['error']

train_range = range(17, 18, 1)
n_runs = 1
dims = [2]

configs = [
    {'weights': [1e-1] * 30,
     'iterations': [5000] * 30},
    {'weights': [1e-2] * 30,
     'iterations': [5000] * 30},
    {'weights': [1e-3] * 30,
     'iterations': [5000] * 30},
    {'weights': [1e-2] * 30,
     'iterations': [5000] * 30},
    {'weights': [1e-2] * 15 + [1e-3] * 15,
     'iterations': [5000] * 30},
    {'weights': [1e-2] * 15 + [1e-1] * 15,
     'iterations': [5000] * 30},
    {'weights': [1e-1] * 15 + [1e-2] * 15,
     'iterations': [5000] * 30},
    {'weights': [1e-3] * 15 + [1e-2] * 15,
     'iterations': [5000] * 30},
    {'weights': [2**j * 1e-4 for j in range(10)],
     'iterations': [15000] * 10},
    {'weights': list(reversed([2**j * 1e-4 for j in range(10)])),
     'iterations': [15000] * 10},
    {'weights': [1.3**j * 1e-4 for j in range(30)],
     'iterations': [5000] * 30},
    {'weights': [1e-2] * 1 + [1e-3] * 26,
     'iterations': [20000] + [5000] * 26},
    {'weights': [1e-2] * 1 + [1e-1] * 26,
     'iterations': [20000] + [5000] * 26},
    {'weights': [1e-2] * 1 + [1.3**j * 1e-4 for j in range(26)],
     'iterations': [20000] + [5000] * 26},
    {'weights': [1e-2] * 1 + [1.3**j * 1e-3 for j in range(26)],
     'iterations': [20000] + [5000] * 26},
]


def mp_worker(data):
    dim, train_seed, i = data

    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]
    lin_constraints = [{'fn': lossfn.norm_linear, 'label': 'det'}]
    config = configs[i]
    constraint_weights = config["weights"]
    iterations = config["iterations"]

    for n_train in train_range:
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'round2_aug_lag_rnd_norm/'
        checkpoint_dir += 'dim-{}_config-{}_ntrain-{}_seed-{}/'.format(dim, i, n_train, train_seed)
        run_experiment_augmented_lagrangian(dim, n_train, train_seed, loss_fn, lin_constraints, constraint_weights,
                                            checkpoint_dir, iterations=iterations)


def mp_handler():
    for dim in dims:
        for i in range(len(configs)):
            for train_seed in range(1683, 1683 + n_runs, 1):
                p = multiprocessing.Process(target=mp_worker, args=((dim, train_seed, i),))
                p.start()


if __name__ == '__main__':
    mp_handler()
