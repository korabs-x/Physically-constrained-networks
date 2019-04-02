from solver import Solver
from model import Net
from dataset import RotationDataset
import lossfn
from torch.utils.data import DataLoader
import torch
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dim", default=0, type=int)
parser.add_argument("--ntrain", default=0, type=int)
parser.add_argument("--lossfn", default='', type=str)
parser.add_argument("--seed", default=1, type=int)

args = parser.parse_args()

SEED_TEST = 0


def get_data_loader(dim, n, seed=SEED_TEST, shuffle=False, batch_size=512):
    dataset = RotationDataset(dim, n, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    args = parser.parse_args()
    dim = args.dim
    n_train = args.ntrain
    train_seed = args.seed
    train_loader = get_data_loader(dim, n_train, seed=train_seed, shuffle=True, batch_size=512)
    test_loader = get_data_loader(dim, 512, seed=SEED_TEST, shuffle=False, batch_size=512)
    model = Net(dim, n_hidden_layers=max(1, int(math.log(dim, 2))))
    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'}]
    if args.lossfn == 'det' or args.lossfn == 'detnorm':
        loss_fn.append({'loss_fn': lossfn.get_det_loss(), 'weight': 0.5, 'label': 'det'})
    if args.lossfn == 'norm' or args.lossfn == 'detnorm':
        loss_fn.append({'loss_fn': lossfn.get_norm_loss(), 'weight': 0.5, 'label': 'norm'})
    solver = Solver(model, loss_fn_train=loss_fn,
                    checkpoint_dir='checkpoints_dim-{}_ntrain-{}_lossfn-{}_seed-{}/'.format(args.dim, args.ntrain,
                                                                                            args.lossfn, args.seed))
    solver.train(train_loader, iterations=20000, test_every_iterations=200, test_loader=test_loader)
