#!/usr/bin/env python3
#SBATCH --gpus=132
#SBATCH --mem=12GB
#SBATCH --time=120:00:00
#SBATCH --mail-user=abstreik
#SBATCH --mail-type=ALL

import lossfn
from experiment import run_experiment
import os
import multiprocessing

n_runs = 20


def mp_worker(data):
    dim, weight_decay, n_train = data
    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]

    for train_seed in range(1683, 1683 + n_runs, 1):
        checkpoint_dir = 'checkpoints/'
        checkpoint_dir += 'round2_weight_decay2/'
        checkpoint_dir += 'dim-{}_ntrain-{}_weightdecay-{}_seed-{}/'.format(dim, n_train, weight_decay, train_seed)
        run_experiment(dim, n_train, train_seed, loss_fn, weight_decay, checkpoint_dir, lr=5e-5, iterations=50000,
                       n_test=4096)


def mp_handler():
    for dim in [2, 3]:
        weight_decays = [0.99, 0.9, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0] if dim == 3 else [0.0001, 0.00001, 0.000001]
        for weight_decay in weight_decays:
            train_range = range(10, 21, 1)

            train_range_spec = train_range
            if dim == 2:
                train_range_spec = range(10, 21, 1)
            if dim == 3:
                train_range_spec = range(20, 201, 20)

            for n_train in train_range_spec:
                p = multiprocessing.Process(target=mp_worker, args=((dim, weight_decay, n_train),))
                p.start()


if __name__ == '__main__':
    mp_handler()
