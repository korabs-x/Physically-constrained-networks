import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self, dim, n_hidden_layers=1, n_hidden_nodes=None):
        super().__init__()
        torch.manual_seed(0)
        self.dim = dim
        hidden_nodes = 50 if n_hidden_nodes is None else n_hidden_nodes
        def create_modules():
            modules = [
                nn.Linear(self.dim - 1, hidden_nodes),
                nn.Sigmoid(),
            ]
            for _ in range(1, n_hidden_layers):
                modules.append(nn.Linear(hidden_nodes, hidden_nodes))
                modules.append(nn.Sigmoid())
            modules.append(nn.Linear(hidden_nodes, 1))
            return modules

        self.ffs = nn.ModuleList([
            nn.Sequential(*create_modules())
            for _ in range(dim * dim)
        ])

    def forward(self, x_batch):
        matrix_entries = [ff(x_batch) for ff in self.ffs]
        return torch.stack(matrix_entries, 1).view((x_batch.shape[0], self.dim, self.dim))



class NetConnected(nn.Module):
    def __init__(self, dim, n_hidden_nodes=None):
        super().__init__()
        torch.manual_seed(0)
        self.dim = dim
        hidden_nodes = 50 if n_hidden_nodes is None else n_hidden_nodes

        self.ff = nn.Sequential(
            nn.Linear(self.dim - 1, hidden_nodes),
            nn.Sigmoid(),
            nn.Linear(hidden_nodes, self.dim * self.dim)
        )

    def forward(self, x_batch):
        return self.ff(x_batch).view((x_batch.shape[0], self.dim, self.dim))

