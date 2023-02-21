import numpy as np
import torch
from torch import nn

class MinNet(nn.Module):
    def __init__(self, n_steps=1):
        super(MinNet, self).__init__()
        
        # Minimal Network Weights
        w1 = torch.tensor([[[[1.0, 1.0, 1.0],[1.0, 0.1, 1.0],[1.0, 1.0, 1.0]]],
                            [[[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]]])
        b1 = torch.tensor([-3.0, -2.0])
        
        w2 = torch.tensor([[[[-10.0]], [[1.0]]]])
        b2 = torch.tensor([0.0])
        
        s=20
        w3 = torch.tensor([[[[2.0 * s]]]])
        b3 = torch.tensor([-1.0 * s])
        
        self.conv_layer1 = nn.Conv2d(1, 2, 3, padding=1, padding_mode='circular')
        self.conv_layer1.weight = torch.nn.Parameter(w1, requires_grad=False)
        self.conv_layer1.bias = torch.nn.Parameter(b1, requires_grad=False)
        self.relu1 = nn.ReLU()
        
        self.conv_layer2 = nn.Conv2d(2, 1, 1)
        self.conv_layer2.weight = torch.nn.Parameter(w2, requires_grad=False)
        self.conv_layer2.bias = torch.nn.Parameter(b2, requires_grad=False)
        self.relu2 = nn.ReLU()
        
        self.conv_layer3 = nn.Conv2d(1, 1, 1)
        self.conv_layer3.weight = torch.nn.Parameter(w3, requires_grad=False)
        self.conv_layer3.bias = torch.nn.Parameter(b3, requires_grad=False)
        self.sigmoid3 = nn.Sigmoid()
        
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu1(out)
        
        out = self.conv_layer2(out)
        out = self.relu2(out)
        
        out = self.conv_layer3(out)
        out = self.sigmoid3(out)
        return out