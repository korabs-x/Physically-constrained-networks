#!/usr/bin/env python3
#SBATCH --gpus=150
#SBATCH --mem=12GB
#SBATCH --time=39:00:00
#SBATCH --mail-user=abstreik
#SBATCH --mail-type=ALL

import os
import multiprocessing
import lossfn
from experiment import run_experiment_variable_loss

loss_fns = ['error', 'det', 'norm', 'detnorm', 'det_variable', 'norm_variable', 'detnorm_variable']# ['error', 'det', 'norm', 'detnorm', 'det_variable', 'norm_variable', 'detnorm_variable']

dim = 2
train_range = range(1, 31, 1)
# n_train = 10
n_runs = 1

def mp_worker(data):
    lossfnstr, n_train = data
    train_seed = 1683
    n_trains = [n_train]
    train_seeds = [train_seed]
    lossfns = [{'loss_fn': [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}], 'iterations': 1e8}]
    if lossfnstr == 'det' or lossfnstr == 'detnorm' or lossfnstr == 'det_variable' or lossfnstr == 'detnorm_variable':
        lossfns[0]['loss_fn'].append({'loss_fn': lossfn.get_det_loss(), 'weight': 0.2, 'label': 'det'})
    if lossfnstr == 'norm' or lossfnstr == 'detnorm' or lossfnstr == 'norm_variable' or lossfnstr == 'detnorm_variable':
        lossfns[0]['loss_fn'].append({'loss_fn': lossfn.get_norm_loss(), 'weight': 0.2, 'label': 'norm'})
    if lossfnstr == 'det_variable' or lossfnstr == 'norm_variable' or lossfnstr == 'detnorm_variable':
        n_trains.append(n_train)
        train_seeds.append(train_seed + 1)
        lossfns[0]['iterations'] = 100
    if lossfnstr == 'det_variable':
        lossfns.append(
            {'loss_fn': [{'loss_fn': lossfn.get_det_loss(), 'weight': 0.2, 'label': 'det'}], 'iterations': 1})
    if lossfnstr == 'norm_variable':
        lossfns.append(
            {'loss_fn': [{'loss_fn': lossfn.get_norm_loss(), 'weight': 0.2, 'label': 'norm'}], 'iterations': 1})
    if lossfnstr == 'detnorm_variable':
        lossfns.append(
            {'loss_fn': [{'loss_fn': lossfn.get_det_loss(), 'weight': 0.2, 'label': 'det'},
                         {'loss_fn': lossfn.get_norm_loss(), 'weight': 0.2, 'label': 'norm'}], 'iterations': 1})

    checkpoint_dir = 'checkpoints/'
    checkpoint_dir += 'checkpoints_semisupervised/'
    checkpoint_dir += 'checkpoint_dim-{}_ntrain-{}_lossfn-{}_seed-{}/'.format(dim, n_train, lossfnstr,
                                                                              train_seed)
    run_experiment_variable_loss(dim, n_trains, train_seeds, lossfns, 0, checkpoint_dir, max_iterations=20000)


def mp_handler():
    for lossfnstr in loss_fns:
        for n_train in train_range:
            p = multiprocessing.Process(target=mp_worker, args=((lossfnstr, n_train),))
            p.start()


if __name__ == '__main__':
    mp_handler()
