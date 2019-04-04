from solver import Solver
from model import Net
from dataset import RotationDataset
import lossfn
from torch.utils.data import DataLoader
import math

SEED_TEST = 0


def get_data_loader(dim, n, seed=SEED_TEST, shuffle=False, batch_size=512):
    dataset = RotationDataset(dim, n, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_experiment(dim, n_train, train_seed, loss_fn, weight_decay, checkpoint_dir):
    train_loader = get_data_loader(dim, n_train, seed=train_seed, shuffle=True, batch_size=512)
    test_loader = get_data_loader(dim, 512, seed=SEED_TEST, shuffle=False, batch_size=512)
    model = Net(dim, n_hidden_layers=max(1, int(math.log(dim, 2))))
    optim_args = {'lr': 1e-3, 'weight_decay': weight_decay}
    solver = Solver(model, loss_fn_train=loss_fn, optim_args=optim_args, checkpoint_dir=checkpoint_dir)
    solver.train(train_loader, iterations=20000, test_every_iterations=200, test_loader=test_loader)
