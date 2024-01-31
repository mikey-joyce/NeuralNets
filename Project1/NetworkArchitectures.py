import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class RBF(nn.Module):
    def __init__(self, K, n_outs):
        super(RBF, self).__init__()
        self.K = K
        self.centers = nn.Parameter(torch.zeros(K))
        self.sigma = nn.Parameter(torch.ones(K))
        self.weights = nn.Linear(K, n_outs)

    def gaussian(self, x):
        return torch.exp(-0.5 * ((x.unsqueeze(1) - self.centers)/self.sigma) ** 2)

    def forward(self, x):
        return self.linear(self.gaussian(x))
