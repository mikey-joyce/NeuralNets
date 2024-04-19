import numpy as np
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, n_in, n_width, n_layers, n_out, mimo):
        super(RNN, self).__init__()
        self.mimo = mimo
        self.n_width = n_width
        self.n_layers = n_layers
        self.rnn = nn.RNN(n_in, n_width, n_layers, batch_first=False, nonlinearity='relu', bias=True)
        self.perceptron = nn.Linear(n_width, n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = torch.zeros(self.n_layers, x.size(0), self.n_width)
        out, _ = self.rnn(x, hidden)

        if self.mimo:
            out = self.sigmoid(self.perceptron(out))
        else:
            out = self.sigmoid(self.perceptron(out[:, -1, :]))

        return out


class LSTM(nn.Module):
    def __init__(self, n_in, n_width, n_layers, n_out, mimo):
        super(LSTM, self).__init__()
        self.mimo = mimo
        self.n_width = n_width
        self.n_layers = n_layers
        self.lstm = nn.LSTM(n_in, n_width, n_layers, batch_first=False, bias=True)
        self.perceptron = nn.Linear(n_width, n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = torch.zeros(self.n_layers, x.size(0), self.n_width)
        cell = torch.zeros(self.n_layers, x.size(0), self.n_width)
        out, _ = self.lstm(x, (hidden, cell))

        if self.mimo:
            out = self.sigmoid(self.perceptron(out))
        else:
            out = self.sigmoid(self.perceptron(out[:, -1, :]))

        return out
