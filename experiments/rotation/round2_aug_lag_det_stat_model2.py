#!/usr/bin/env python3
#SBATCH --gpus=11
#SBATCH --mem=12GB
#SBATCH --time=120:00:00
#SBATCH --mail-user=abstreik
#SBATCH --mail-type=ALL

import os
import multiprocessing
import lossfn
from experiment import run_experiment_augmented_lagrangian_auto

loss_fns = ['error']

train_range = range(10, 21, 1)
n_runs = 20
dims = [2]

config = {'constraint_sq_weight': 0.001,
          'constraint_sq_weight_multiplier': 1.05,
          'eps': 1e-3,
          'gam': 0.1,
          'eps_gam_decay_rate': 0.95,
          'grad_threshold': None,
          'iterations': 50}


def mp_worker(data):
    dim, n_train = data

    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]
    lin_constraints = [{'fn': lossfn.det_linear, 'label': 'det'}]

    for train_seed in range(1683, 1683 + n_runs, 1):
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'round2_aug_lag_det_model2/'
        checkpoint_dir += 'sqweight-{}_sqwmul-{}_eps-{}_gam-{}_decrate-{}_gradthresh-{}_it-{}/'.format(
            config['constraint_sq_weight'], config['constraint_sq_weight_multiplier'], config['eps'], config['gam'],
            config['eps_gam_decay_rate'], config['grad_threshold'], config['iterations'])
        checkpoint_dir += 'dim-{}_ntrain-{}_seed-{}/'.format(dim, n_train, train_seed)

        run_experiment_augmented_lagrangian_auto(dim, n_train, train_seed, loss_fn, lin_constraints,
                                                 config['constraint_sq_weight'],
                                                 config['constraint_sq_weight_multiplier'],
                                                 config['eps'],
                                                 config['gam'],
                                                 config['eps_gam_decay_rate'],
                                                 config['grad_threshold'],
                                                 checkpoint_dir,
                                                 iterations=config['iterations'],
                                                 model_class=NetConnected)


def mp_handler():
    for dim in dims:
        for n_train in train_range:
            p = multiprocessing.Process(target=mp_worker, args=((dim, n_train),))
            p.start()


if __name__ == '__main__':
    mp_handler()
