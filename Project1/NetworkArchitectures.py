import numpy as np
import torch
import torch.nn as nn
from FCM import FCM


class RBF(nn.Module):
    def __init__(self, k: int, data: np.ndarray, sigma: float = 1.0, mode: int = 0, n_perceptrons=1):
        super(RBF, self).__init__()
        self.K = k

        self.mode = mode
        if self.mode == 0:    # one sigma value
            self.covar = nn.Parameter(torch.ones(1, dtype=torch.float64) * sigma, requires_grad=True)
        elif self.mode == 1:  # one sigma value for each RBF node
            self.covar = nn.Parameter(torch.ones(self.K, dtype=torch.float64) * sigma, requires_grad=True)
        elif self.mode == 2:  # full covariance
            self.covar = nn.Parameter(torch.rand((self.K, data.shape[1], data.shape[1]), dtype=torch.float64) * sigma, requires_grad=True)
        else:
            raise Exception('Covariance mode does not exist; pick mode one (0), K (1), or full-covariance (2)')

        # initialize centers of RBF utilizing FCM :)
        fcm = FCM(data, self.K)
        centers, membership, fpc = fcm.fit()

        self.centers = nn.Parameter(torch.from_numpy(centers), requires_grad=False)
        self.weights = nn.Parameter(((torch.rand(n_perceptrons, self.K, dtype=torch.float64) - 0.5) * 2.0) * (1.0/self.K), requires_grad=True)
        self.bias = nn.Parameter(((torch.rand(n_perceptrons, dtype=torch.float64) - 0.5) * 2.0) * (1.0/self.K), requires_grad=True)

    def rbf(self, x):
        if self.mode == 2:
            diffs = x.unsqueeze(1) - self.centers.unsqueeze(0)
            covar_inv = torch.inverse(self.covar)
            exp_values = -0.5*torch.matmul(torch.matmul(diffs.unsqueeze(2), covar_inv), diffs.unsqueeze(-1)).squeeze()
            return torch.exp(exp_values)
        else:
            return torch.exp(-torch.pow(torch.tensor(x, dtype=torch.float64).unsqueeze(1) - self.centers, 2).sum(dim=2) / (2 * self.covar**2))

    def forward(self, x):
        g = self.rbf(x)
        if self.K == 1:
            perceptron = g * self.weights
        else:
            perceptron = torch.matmul(g, self.weights.T)
        out = torch.sigmoid(perceptron + self.bias)
        return torch.where(torch.isnan(out), torch.tensor(0.0), out)  # force a solution to numerical instability

