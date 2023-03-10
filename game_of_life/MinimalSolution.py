import numpy as np
import torch
from torch import nn

class MinNet(nn.Module):
    def __init__(self, n_steps=1):
        super(MinNet, self).__init__()
        
        self.num_steps = n_steps
        self.overcompleteness_factor = 1
        
        # Minimal Network Weights
        w1 = torch.tensor([[[[1.0, 1.0, 1.0],[1.0, 0.1, 1.0],[1.0, 1.0, 1.0]]],
                            [[[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]]])
        b1 = torch.tensor([-3.0, -2.0])
        
        w2 = torch.tensor([[[[-10.0]], [[1.0]]]])
        b2 = torch.tensor([0.0])
        
        s=20
        w3 = torch.tensor([[[[2.0 * s]]]])
        b3 = torch.tensor([-1.0 * s])
        
        self.type1_layers = nn.ModuleList()
        self.type2_layers = nn.ModuleList()
        
        for i in range(self.num_steps):
            type1_layer = nn.Conv2d(1, 2, 3, padding=1, padding_mode='circular')
            type1_layer.weight = torch.nn.Parameter(w1, requires_grad=False)
            type1_layer.bias = torch.nn.Parameter(b1, requires_grad=False)
            self.type1_layers.append(type1_layer)
        
            type2_layer = nn.Conv2d(2, 1, 1)
            type2_layer.weight = torch.nn.Parameter(w2, requires_grad=False)
            type2_layer.bias = torch.nn.Parameter(b2, requires_grad=False)
            self.type2_layers.append(type2_layer)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.final_conv_layer = nn.Conv2d(1, 1, 1)
        self.final_conv_layer.weight = torch.nn.Parameter(w3, requires_grad=False)
        self.final_conv_layer.bias = torch.nn.Parameter(b3, requires_grad=False)
        self.sigmoid3 = nn.Sigmoid()
        
    def forward(self, x):
        out=x
        
        for i in range(self.num_steps):
            out = self.type1_layers[i](out)
            out = self.relu1(out)

            out = self.type2_layers[i](out)
            out = self.relu2(out)
        
        out = self.final_conv_layer(out)
        out = self.sigmoid3(out)
        return out
    
    '''
    This function is meant for inspecting the intermediary results of the
    model. That is, it produces the output after n pairs of intermediary
    convolutional layers have been passed which corresponds to how the
    model "simulates" n steps of the game of Life.
    '''
    def inspect_step(self, x, n):
        
        if(n > self.num_steps):
            print(f'Error: {n} is more steps than the model has layers for.')
        else:
            out=x
            
            for i in range(n):
                out = self.type1_layers[i](out)
                out = self.relu1(out)

                out = self.type2_layers[i](out)
                out = self.relu2(out)
                
            out = self.final_conv_layer(out)
            out = self.sigmoid3(out)
            return out