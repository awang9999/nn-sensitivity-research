import numpy as np
import torch
from torch import nn

class OPNet(nn.Module):
    '''
    - m refers to the factor of overcompleteness of parameters in the CNN
    - n  refers to the number of steps of the Game of Life this CNN simulates.
    This is consistent with the convention used in the Kenyon and Springer paper
    about learning the Game of Life.
    '''
    def __init__(self, m=2, n=1):
        super(OPNet, self).__init__()
        
        self.overcompleteness_factor = m
        self.num_steps = n
        
        self.type1_layers = nn.ModuleList()
        self.type2_layers = nn.ModuleList()
        
        for i in range(self.num_steps):
            type1_layer = nn.Conv2d(m, m*2, 3, padding=1, padding_mode='circular')
            self.type1_layers.append(type1_layer)
        
            type2_layer = nn.Conv2d(m*2, m, 1)
            self.type2_layers.append(type2_layer)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.final_conv_layer = nn.Conv2d(m, 1, 1)
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