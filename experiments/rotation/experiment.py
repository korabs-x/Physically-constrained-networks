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


def run_experiment_variable_loss(dim, n_trains, train_seeds, loss_fns, weight_decay, checkpoint_dir,
                                 max_iterations=20000):
    train_loaders = [get_data_loader(dim, n_train, seed=train_seed, shuffle=False, batch_size=512) for
                     n_train, train_seed in zip(n_trains, train_seeds)]
    test_loader = get_data_loader(dim, 512, seed=SEED_TEST, shuffle=False, batch_size=512)
    model = Net(dim, n_hidden_layers=max(1, int(math.log(dim, 2))))
    optim_args = {'lr': 1e-3, 'weight_decay': weight_decay}
    solver = Solver(model, optim_args=optim_args, checkpoint_dir=checkpoint_dir)
    iterations = 0
    loss_index = 0
    while iterations < max_iterations:
        train_loader = train_loaders[loss_index]
        solver.set_loss_fn_train(loss_fns[loss_index]["loss_fn"])
        n_iterations_left = max_iterations - iterations
        n_iterations = min(loss_fns[loss_index]["iterations"], n_iterations_left)
        solver.train(train_loader, iterations=n_iterations,
                     test_every_iterations=200, test_loader=test_loader, save_final=(n_iterations_left == n_iterations))
        if loss_index == 0:
            iterations += n_iterations
        loss_index = (loss_index + 1) % len(loss_fns)
