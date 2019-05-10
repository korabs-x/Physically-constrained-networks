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
                          'weight': constraint_sq_weight * 0.5, 'label': constraint_label + '_sq'},
                         {'loss_fn': lossfn.get_constrained_loss_linear(constraint_fn, constraint_ln_weights[i]),
                          'weight': -1, 'label': constraint_label + '_lin'}]
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


def run_experiment_augmented_lagrangian_auto(dim, n_train, train_seed, loss_fn, lin_constraints, constraint_sq_weight,
                                             constraint_sq_weight_multiplier, eps, gam, eps_gam_decay_rate,
                                             grad_threshold, checkpoint_dir, iterations=100, lr=5e-5, n_test=4096):
    train_loader = get_data_loader(dim, n_train, seed=train_seed, shuffle=False, batch_size=512)
    test_loader = get_data_loader(dim, n_test, seed=SEED_TEST, shuffle=False, batch_size=min(n_test, 8 * 512))
    model = Net(dim, n_hidden_layers=max(1, int(math.log(dim, 2))))
    optim_args = {'lr': lr}
    solver = Solver(model,
                    loss_fn_train=loss_fn,
                    optim_args=optim_args,
                    checkpoint_dir=checkpoint_dir)

    # augmented lagrangian parameters
    initial_ln_weight = 0
    constraint_ln_weights = [torch.ones(size=(n_train,)) * initial_ln_weight for _ in lin_constraints]
    # constraint_sq_weight = 1e-2
    # constraint_sq_weight_multiplier = 1

    # eps and gam are used to determine the threshold when to end the iteration
    # eps = 1e-3
    # gam = 10
    # eps_gam_decay_rate = 0.98
    grad_norm_threshold = eps if grad_threshold is None else grad_threshold

    total_iterations = 0

    for iteration in range(iterations):
        print("Start iteration {}, after {} iterations, sqweight={}".format(iteration, total_iterations, constraint_sq_weight))
        loss_fns = [] + loss_fn
        for i, constraint_info in enumerate(lin_constraints):
            constraint_fn = constraint_info["fn"]
            constraint_label = constraint_info["label"]
            loss_fns += [{'loss_fn': lossfn.get_constrained_loss_quadratic(constraint_fn),
                          'weight': constraint_sq_weight * 0.5, 'label': constraint_label + '_sq'},
                         {'loss_fn': lossfn.get_constrained_loss_linear(constraint_fn, constraint_ln_weights[i]),
                          'weight': -1, 'label': constraint_label + '_lin'}]
        solver.set_loss_fn_train(loss_fns)

        # iterate until gradient norm is smaller than grad_norm_threshold
        for round in range(100):
            solver.train(train_loader, iterations=500, test_every_iterations=500, test_loader=test_loader,
                         save_final=False, prints=False)
            total_iterations += 500
            print("Test loss {}".format(solver.hist["test_loss"][-1]))
            print("Gradient norm       {}".format(solver.hist["gradient_norm"][-1]))
            print("grad_norm_threshold {}".format(grad_norm_threshold))
            if solver.hist["gradient_norm"][-1] <= grad_norm_threshold and round >= 5:
                break

        # update weights
        constraint_vals = [[] for _ in lin_constraints]
        for (points, angles, points_rotated) in train_loader:
            output_matrix, prediction = solver.forward(angles, points)
            for i in range(len(lin_constraints)):
                constraint_vals[i].append(lin_constraints[i]["fn"](prediction, points_rotated, output_matrix).detach())
        constraint_vals = [torch.stack(val_list).view(-1) for val_list in constraint_vals]
        for i in range(len(lin_constraints)):
            constraint_ln_weights[i] -= constraint_sq_weight * constraint_vals[i]

        total_constraint_norm = torch.stack(constraint_vals).norm(2)
        if grad_threshold is None:
            print("Set grad_threshold to min({}, {} * {})".format(eps, total_constraint_norm, gam))
            print("Set grad_threshold to min({}, {})".format(eps, total_constraint_norm * gam))
            grad_norm_threshold = min(eps, total_constraint_norm * gam)
        eps *= eps_gam_decay_rate
        gam *= eps_gam_decay_rate
        constraint_sq_weight *= constraint_sq_weight_multiplier

    solver.train(train_loader, iterations=0, test_loader=test_loader, save_final=True)
