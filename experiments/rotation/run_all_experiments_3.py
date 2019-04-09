#!/usr/bin/env python3
#SBATCH --gpus=44
#SBATCH --mem=12GB
#SBATCH --time=120:00:00
#SBATCH --mail-user=abstreik
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

import os
import multiprocessing

loss_fns = ['error', 'det', 'norm', 'detnorm']

dim = 3
if dim == 2:
    train_range = range(1, 51, 1)
elif dim == 3:
    train_range = range(25, 251, 25)
elif dim == 5:
    train_range = range(50, 5001, 50)
elif dim == 10:
    train_range = list(range(2500, 10000, 2500)) + list(range(10000, 30000, 5000)) + list(range(30000, 500001, 10000))


def mp_worker(data):
    index, dimension, lossfn, ntrain = data
    run = 'python3 run_experiment2.py --dim {} --ntrain {} --lossfn {} --seed {}'.format(dimension, ntrain, lossfn,
                                                                                        1683)
    print(run)
    os.system(run)


def mp_handler():
    for i, lossfn in enumerate(loss_fns):
        for ntrain in train_range:
            p = multiprocessing.Process(target=mp_worker, args=((i, dim, lossfn, ntrain),))
            p.start()


if __name__ == '__main__':
    mp_handler()
