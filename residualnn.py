import torch
import torch.nn as nn
import numpy as np

class ResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(ResidualNetwork, self).__init__()
        self.layers = []

        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())

        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
        