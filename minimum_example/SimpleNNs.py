import os
import torch
from torch import nn
from torch.utils.data import DataLoader

class OneNet(nn.Module):
    def __init__(self, input_n: int, output_n: int, hidden_n: int):
        super(OneNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_n, hidden_n),
            nn.ReLU(),
            nn.Linear(hidden_n, output_n)
        )

    def forward(self, x):
        x = self.flatten(x)
        yhat = self.linear_relu_stack(x)
        return yhat
    
class TwoNet(nn.Module):
    def __init__(self, input_n: int, output_n: int, hidden_n: int):
        super(TwoNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_n, hidden_n),
            nn.ReLU(),
            nn.Linear(hidden_n, hidden_n),
            nn.ReLU(),
            nn.Linear(hidden_n, output_n)
        )

    def forward(self, x):
        x = self.flatten(x)
        yhat = self.linear_relu_stack(x)
        return yhat