import torch.nn as nn
import torch


class NetNomatrix(nn.Module):
    def __init__(self, dim, n_hidden_nodes=None):
        super().__init__()
        torch.manual_seed(0)
        self.dim = dim
        hidden_nodes = 50 if n_hidden_nodes is None else n_hidden_nodes

        self.ff = nn.Sequential(
            nn.Linear(2 * self.dim - 1, hidden_nodes),
            nn.Sigmoid(),
            nn.Linear(hidden_nodes, self.dim)
        )

    def forward(self, x_batch):
        return self.ff(x_batch)

