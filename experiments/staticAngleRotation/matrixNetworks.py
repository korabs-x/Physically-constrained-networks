
# coding: utf-8

# In[91]:

import pandas as pd
import torch.nn as nn
import random
from torch.optim import Adam
from torch.utils.data import DataLoader,TensorDataset
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import RotationMatrix


# In[92]:

points = pd.read_csv("rotated_points_angle.csv")


# In[93]:

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_nodes = 5
        self.ff = nn.Sequential(
            nn.Linear(1, hidden_nodes),
            nn.Sigmoid(),
            nn.Linear(hidden_nodes, 1),
        )
    
    def forward(self, x_batch):
        return self.ff(x_batch)


# ## Check if model can learn sine function

# In[132]:

def train_sin():
    random.seed(234)
    model = Net()
    opt = Adam(model.parameters())
    model.train()
    for i in tqdm(range(300000)):
        alpha = random.uniform(0, 2*math.pi)
        pred = model(torch.Tensor([alpha]))
        y_true = math.sin(alpha)
        opt.zero_grad()
        loss = (pred - y_true) ** 2
        loss.backward()
        opt.step()
    return model


# In[133]:

def plot_sin_model(model):
    test = np.linspace(0, 2*math.pi, 100)
    model.eval()
    pred = model(torch.Tensor(np.matrix(test).transpose()))
    plt.plot(test, np.sin(test))
    plt.plot(test, pred.detach().numpy())
    plt.show()


# In[134]:

# plot_sin_model(train_sin())


# # Train nets for matrix

# In[94]:

def get_loader(data):
    X = torch.FloatTensor(data[["x1", "y1", "alpha"]].values)
    y = torch.FloatTensor(data[["x2", "y2"]].values)
    torch_data = TensorDataset(X, y)
    loader = DataLoader(torch_data, shuffle=False)
    return loader


# In[95]:

def calc_pred_loss(x_batch, y_batch, nets):
    alphas = x_batch[:, [2]]
    matrix_entries = [model(alphas) for model in nets]
    loss = 0
    preds = []
    for i in range(len(alphas)):
        predx = matrix_entries[0][i] * x_batch[i][0] + matrix_entries[1][i] * x_batch[i][1]
        predy = matrix_entries[2][i] * x_batch[i][0] + matrix_entries[3][i] * x_batch[i][1]
        loss += (predx - y_batch[i][0]) ** 2
        loss += (predy - y_batch[i][1]) ** 2
        preds.append((predx.item(), predy.item()))
    loss /= len(alphas)
    return {"loss": loss, "preds": preds}


# In[96]:

def calc_loss(x_batch, y_batch, nets):
    return calc_pred_loss(x_batch, y_batch, nets)["loss"]


# In[137]:

def train(n_points_train_val):
    n_points = round(n_points_train_val * 0.8)
    train_loader = get_loader(points.head(n_points))
    n_points_val = max(n_points_train_val - n_points, 1)
    val_loader = get_loader(points.tail(n_points_val))

    nets = [Net() for _ in range(4)]
    opts = [Adam(model.parameters()) for model in nets]
    val_losses = []
    train_losses = []
    epochs = []
    epoch = 0
    epochs_per_validation = 100
    while True:
        # training
        for model in nets:
            model.train()
        for x_batch, y_batch in train_loader:
            loss = calc_loss(x_batch, y_batch, nets)
            for opt in opts:
                opt.zero_grad()
            loss.backward()
            for opt in opts:
                opt.step()
        # validation
        if epoch % epochs_per_validation == 0:
            for model in nets:
                model.eval()
            val_loss = 0
            train_loss = 0
            for x_batch, y_batch in val_loader:
                val_loss += calc_loss(x_batch, y_batch, nets)
            for x_batch, y_batch in train_loader:
                train_loss += calc_loss(x_batch, y_batch, nets)
            val_loss /= n_points_val
            train_loss /= n_points
            val_losses.append(val_loss.item())
            train_losses.append(train_loss.item())
            epochs.append(epoch)
            # if epoch % 1000 == 0:
            #     print("Epoch {}: Val {}, Train {}".format(epoch, val_losses[-1], train_losses[-1]))
            reference_loss_index = 500//epochs_per_validation
            if epoch > 1000 and len(val_losses) > reference_loss_index and val_losses[-1] > val_losses[-reference_loss_index]:
                break
        epoch += 1
    return {"nets": nets, "train_loss": train_losses, "val_loss": val_losses, "epochs": epochs}


# In[138]:

# build test set
n_test = 16
test = []
for alpha in np.linspace(0, 2*math.pi, n_test)[:-1]:
    x1, y1 = RotationMatrix.apply_rotation(1, 0, alpha)
    x2, y2 = RotationMatrix.apply_rotation(x1, y1, alpha)
    test.append([alpha, x1, y1, x2, y2])
test = pd.DataFrame(test, columns=['alpha', 'x1', 'y1', 'x2', 'y2'])
test_loader = get_loader(test)


# In[139]:

def test(nets):
    test_loss = 0
    test_preds = []
    for x_batch, y_batch in test_loader:
        result = calc_pred_loss(x_batch, y_batch, nets)
        test_preds += result["preds"]
        test_loss += result["loss"]
    test_loss /= len(test_loader)
    return {"loss": test_loss.item(), "preds": test_preds}


# In[140]:

def train_test(n_points_train_val, visualize=True):
    train_result = train(n_points_train_val)
    test_result = test(train_result["nets"])
    if visualize:
        plt.plot(train_result["epochs"], train_result["val_loss"])
        plt.plot(train_result["epochs"], train_result["train_loss"])
        plt.show()
        print("Test loss = {}".format(test_result["loss"]))
    return train_result, test_result


# In[143]:

results = []
for n_points_train_val in range(1, 3):
    train_result, test_result = train_test(n_points_train_val, visualize=False)
    del train_result["nets"]
    results.append([n_points_train_val, str(train_result), str(test_result)])
df = pd.DataFrame(results, columns=["n_points_train_val", "train_result", "test_result"])
df.to_csv("train_limited_data_results.csv", index=False)


# In[145]:

#import os
#os.system('say "your program has finished"')


# In[ ]:



