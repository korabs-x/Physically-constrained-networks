# Physically constrained networks

This repository contains all of the implementation and results of the Bachelor's thesis "Physically constrained networks" written at the Technical University of Munich. The thesis can be found at [TUM_Thesis/main.pdf](TUM_Thesis/main.pdf).

### Summary:
We incorporate physical constraints into deep learning models to improve performance in terms of test loss and physical feasibility of the predictions. We apply the Penalty Method, the Augmented Lagrangian Method and Physical Projection to the problem of learning a rotation. Physical constraints include norm preservation and a determinant of one of the rotation matrix. The first two methods achieve significant performance improvements and lower physical constraint violations, while the Physical Projection method does not seem to be helpful.

## Implementation

Requirements:
- Python >= 3.5
- PyTorch >= 1.0
- numpy
- matplotlib (optional for plots)

### Details about the files
- dataset.py: Get the rotation data as a PyTorch dataset using RotationDataset(dimension, n_datapoints, seed=random_seed)
- coordinate_transformation.py: apply_euclidean_rotation(point, angles) returns the rotated point. Only used by dataset.py
- model.py: Get Model 2 (architecture is explained in the thesis) with NetConnected100(dimension), Model 3 with Net(dimension)
- model_nomatrix3.py: Get Model 1 with NetNomatrix16V2(dimension)
- lossfn.py: Several functions returning loss functions, e.g. get_mse_loss
- solver.py: Functions for training and testing a model and to save and load checkpoints
- experiment.py: Functions that were used for calculating experiments for the thesis. Includes functions to execute the Augmented Lagrangian Method.
- run_*.py: Files that calculate results for different hyperparameters for direct comparison. These were the only files executed to calculate the results of the thesis.
- plots.ipynb: Contains the code to reproduce all plots included in the thesis. Check functions used in this file to see how to access results and calculate e.g. Physical Loss of a trained model on test data
- checkpoints/: Contains all calculation results in the form of model states

### Reproduce results
An example to calculate the results of applying the Penalty Method on the determinant constraint with different physical loss weights is given in the file experiments/rotation/run_det_weights.py. We additionally give a simple example to train a model with a determinant loss weight of 0 (baseline) and 0.1 for a single seed with 12 training data points. We assume that the code is executed in the directory experiments/rotation/:
```
import lossfn
from experiment import run_experiment
# set parameters
dim = 2
n_train = 12
train_seed = 1683
det_weights = [0, 0.1]
# set a directory where the best and final model state is saved
# note that the saved model state also includes test loss values
checkpoint_dir = 'checkpoints/'
checkpoint_dir += 'example_penalty_determinant/'
# run the experiment
for det_weight in det_weights:
    # set the training loss functions
    # the total training loss is the sum of these loss functions weighted with the respective weight
    loss_fn = [{'loss_fn': lossfn.get_mse_loss(), 'weight': 1, 'label': 'mse'},
    {'loss_fn': lossfn.get_det_loss(), 'weight': det_weight, 'label': 'det'}]
    check_dir_spec = checkpoint_dir + 'dim-{}_detweight-{}_ntrain-{}_seed-{}/'.format(dim, det_weight, n_train, train_seed)
    run_experiment(dim, n_train, train_seed, loss_fn, 0, check_dir_spec, lr=5e-5, iterations=50000, n_test=4096)
```
In order to check the results, execute the following code:
```
from model import Net
from solver import Solver
solver = Solver(Net(dim))
for det_weight in det_weights:
    check_dir_spec = checkpoint_dir + 'dim-{}_detweight-{}_ntrain-{}_seed-{}/'.format(dim, det_weight, n_train, train_seed)
    solver.load_checkpoint(check_dir_spec + 'final.pkl')
    # solver.hist now contains all stats captured throughout the training process
    print("Final test loss for det_weight = {} is {}".format(det_weight, round(solver.hist["test_loss"][-1], 4)))
```
Output: 
```
Final test loss for det_weight = 0 is 0.0196
Final test loss for det_weight = 0.1 is 0.0051
```
