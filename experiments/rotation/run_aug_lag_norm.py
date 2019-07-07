#!/usr/bin/env python3
#SBATCH --gpus=48
#SBATCH --mem=12GB
#SBATCH --time=46:00:00
#SBATCH --mail-user=abstreik
#SBATCH --mail-type=ALL

import os
import multiprocessing
import lossfn
from experiment import run_experiment_augmented_lagrangian_auto

loss_fns = ['error']

train_range = range(10, 21, 4)
n_runs = 2
dims = [2]

configs = [
    {'constraint_sq_weight': 0.1,
     'constraint_sq_weight_multiplier': 1,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': None,
     'iterations': 100},
    {'constraint_sq_weight': 0.01,
     'constraint_sq_weight_multiplier': 1,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': None,
     'iterations': 100},
    {'constraint_sq_weight': 0.01,
     'constraint_sq_weight_multiplier': 1.01,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': None,
     'iterations': 100},
    {'constraint_sq_weight': 0.1,
     'constraint_sq_weight_multiplier': 1.02,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': None,
     'iterations': 100},
    {'constraint_sq_weight': 0.01,
     'constraint_sq_weight_multiplier': 1.05,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': None,
     'iterations': 100},
    {'constraint_sq_weight': 0.0001,
     'constraint_sq_weight_multiplier': 1.1,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': None,
     'iterations': 100},
    {'constraint_sq_weight': 0.001,
     'constraint_sq_weight_multiplier': 1.1,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': None,
     'iterations': 100},
    {'constraint_sq_weight': 1e-5,
     'constraint_sq_weight_multiplier': 2,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': None,
     'iterations': 20},
    {'constraint_sq_weight': 0.1,
     'constraint_sq_weight_multiplier': 1,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': 1e-3,
     'iterations': 100},
    {'constraint_sq_weight': 0.01,
     'constraint_sq_weight_multiplier': 1,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': 1e-3,
     'iterations': 100},
    {'constraint_sq_weight': 0.01,
     'constraint_sq_weight_multiplier': 1.01,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': 1e-3,
     'iterations': 100},
    {'constraint_sq_weight': 0.1,
     'constraint_sq_weight_multiplier': 1.02,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': 1e-3,
     'iterations': 100},
    {'constraint_sq_weight': 0.01,
     'constraint_sq_weight_multiplier': 1.05,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': 1e-3,
     'iterations': 100},
    {'constraint_sq_weight': 0.0001,
     'constraint_sq_weight_multiplier': 1.1,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': 1e-3,
     'iterations': 100},
    {'constraint_sq_weight': 0.001,
     'constraint_sq_weight_multiplier': 1.1,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': 1e-3,
     'iterations': 100},
    {'constraint_sq_weight': 1e-5,
     'constraint_sq_weight_multiplier': 2,
     'eps_gam_decay_rate': 0.98,
     'grad_threshold': 1e-3,
     'iterations': 20},
]


def mp_worker(data):
    dim, n_train, i = data

    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]
    lin_constraints = [{'fn': lossfn.norm_linear, 'label': 'norm'}]
    config = configs[i]

    for train_seed in range(1683, 1683 + n_runs, 1):
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'round2_aug_lag_auto_norm/'
        checkpoint_dir += 'dim-{}_config-{}_ntrain-{}_seed-{}/'.format(dim, i, n_train, train_seed)
        run_experiment_augmented_lagrangian_auto(dim, n_train, train_seed, loss_fn, lin_constraints,
                                                 config['constraint_sq_weight']*100,
                                                 config['constraint_sq_weight_multiplier'],
                                                 config['eps_gam_decay_rate'],
                                                 config['grad_threshold'],
                                                 checkpoint_dir,
                                                 iterations=config['iterations'])


def mp_handler():
    for dim in dims:
        for i in range(len(configs)):
            for n_train in train_range:
                p = multiprocessing.Process(target=mp_worker, args=((dim, n_train, i),))
                p.start()


if __name__ == '__main__':
    mp_handler()
