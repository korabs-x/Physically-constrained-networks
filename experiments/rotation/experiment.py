from solver import Solver
from model import Net
from dataset import RotationDataset
import lossfn
from torch.utils.data import DataLoader
import math
import torch

SEED_TEST = 0


def get_data_loader(dim, n, seed=SEED_TEST, shuffle=False, batch_size=512):
    dataset = RotationDataset(dim, n, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_experiment(dim, n_train, train_seed, loss_fn, weight_decay, checkpoint_dir, fn_matrix=lambda x: x,
                   fn_pred=lambda x: x, iterations=20000, lr=1e-3, n_test=512):
    train_loader = get_data_loader(dim, n_train, seed=train_seed, shuffle=True, batch_size=512)
    test_loader = get_data_loader(dim, n_test, seed=SEED_TEST, shuffle=False, batch_size=min(n_test, 8 * 512))
    model = Net(dim, n_hidden_layers=max(1, int(math.log(dim, 2))))
    optim_args = {'lr': lr, 'weight_decay': weight_decay}
    solver = Solver(model,
                    loss_fn_train=loss_fn,
                    optim_args=optim_args,
                    checkpoint_dir=checkpoint_dir,
                    fn_matrix=fn_matrix,
                    fn_pred=fn_pred)
    solver.train(train_loader, iterations=iterations, test_every_iterations=200, test_loader=test_loader)


def run_experiment_variable_loss(dim, n_trains, train_seeds, loss_fns, weight_decay, checkpoint_dir, lr=1e-3,
                                 max_iterations=20000, n_test=4096):
    train_loaders = [get_data_loader(dim, n_train, seed=train_seed, shuffle=False, batch_size=512) for
                     n_train, train_seed in zip(n_trains, train_seeds)]
    test_loader = get_data_loader(dim, n_test, seed=SEED_TEST, shuffle=False, batch_size=512)
    model = Net(dim, n_hidden_layers=max(1, int(math.log(dim, 2))))
    optim_args = {'lr': lr, 'weight_decay': weight_decay}
    solver = Solver(model, optim_args=optim_args, checkpoint_dir=checkpoint_dir, loss_fn_train=loss_fns[0]["loss_fn"])
    iterations = 0
    loss_index = 0
    while iterations < max_iterations:
        train_loader = train_loaders[loss_index]
        solver.set_loss_fn_train(loss_fns[loss_index]["loss_fn"])
        n_iterations_left = max_iterations - iterations
        n_iterations = min(loss_fns[loss_index]["iterations"], n_iterations_left)
        solver.train(train_loader, iterations=n_iterations,
                     test_every_iterations=500, test_loader=test_loader,
                     save_final=(n_iterations_left == n_iterations))
        # if loss_index == 0:
        iterations += n_iterations
        loss_index = (loss_index + 1) % len(loss_fns)


def run_experiment_augmented_lagrangian(dim, n_train, train_seed, loss_fn, lin_constraints, constraint_weights,
                                        checkpoint_dir, iterations=10000, lr=5e-5, n_test=4096):
    train_loader = get_data_loader(dim, n_train, seed=train_seed, shuffle=False, batch_size=512)
    test_loader = get_data_loader(dim, n_test, seed=SEED_TEST, shuffle=False, batch_size=min(n_test, 8 * 512))
    model = Net(dim, n_hidden_layers=max(1, int(math.log(dim, 2))))
    optim_args = {'lr': lr}
    solver = Solver(model,
                    loss_fn_train=loss_fn,
                    optim_args=optim_args,
                    checkpoint_dir=checkpoint_dir)
    initial_ln_weight = 0
    constraint_ln_weights = [torch.ones(size=(n_train,)) * initial_ln_weight for _ in lin_constraints]
    # constraint_fn = lossfn.det_linear

    for step, constraint_sq_weight in enumerate(constraint_weights):
        n_iterations = iterations if type(iterations) == int else iterations[step]
        loss_fns = [] + loss_fn
        for i, constraint_info in enumerate(lin_constraints):
            constraint_fn = constraint_info["fn"]
            constraint_label = constraint_info["label"]
            loss_fns += [{'loss_fn': lossfn.get_constrained_loss_quadratic(constraint_fn),
                          'weight': constraint_sq_weight * 0.5, 'label': constraint_label+'_sq'},
                         {'loss_fn': lossfn.get_constrained_loss_linear(constraint_fn, constraint_ln_weights[i]),
                          'weight': -1, 'label': constraint_label+'_lin'}]
        solver.set_loss_fn_train(loss_fns)
        solver.train(train_loader, iterations=n_iterations, test_every_iterations=500, test_loader=test_loader,
                     save_final=(step == len(constraint_weights) - 1))
        # update weights
        constraint_vals = [[] for _ in lin_constraints]
        for (points, angles, points_rotated) in train_loader:
            output_matrix, prediction = solver.forward(angles, points)
            for i in range(len(lin_constraints)):
                constraint_vals[i].append(lin_constraints[i]["fn"](prediction, points_rotated, output_matrix).detach())
        constraint_vals = [torch.stack(val_list).view(-1) for val_list in constraint_vals]
        for i in range(len(lin_constraints)):
            constraint_ln_weights[i] -= constraint_sq_weight * constraint_vals[i]


