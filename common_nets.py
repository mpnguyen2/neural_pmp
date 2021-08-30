# All common types of networks

import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, input_dim, output_dim, layer_dims=[], activation='tanh'):
        super(Mlp, self).__init__()
        self.layers = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        if len(layer_dims) != 0:
            self.layers.append(nn.Linear(input_dim, layer_dims[0]))
            for i in range(len(layer_dims)-1):
                if activation == 'tanh':
                    self.layers.append(nn.Tanh())
                self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if activation == 'tanh':
                self.layers.append(nn.Tanh())
            self.layers.append(nn.Linear(layer_dims[-1], output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, output_dim))
        # Composing all layers
        self.net = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.net(x)
