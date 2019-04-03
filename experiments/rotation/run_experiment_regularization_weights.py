#!/usr/bin/env python3
# SBATCH --cpus=4
# SBATCH --mem=12GB
# SBATCH --time=10:00:00
# SBATCH --mail-user=abstreik
# SBATCH --mail-type=ALL

import lossfn
from experiment import run_experiment
import os
import multiprocessing


def mp_worker(data):
    weight_decay = data
    dim = 2
    train_seed = 1683
    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]
    for n_train in train_range:
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'checkpoints_weight_decay_mse/'
        checkpoint_dir += 'checkpoint_dim-{}_ntrain-{}_weightdecay-{}_seed-{}'.format(dim, n_train, weight_decay,
                                                                                      train_seed)
        run_experiment(dim, n_train, train_seed, loss_fn, weight_decay, checkpoint_dir)


def mp_handler():
    weight_decays = []
    for weight_decay in weight_decays:
        p = multiprocessing.Process(target=mp_worker, args=(weight_decay,))
        p.start()


if __name__ == '__main__':
    mp_handler()
