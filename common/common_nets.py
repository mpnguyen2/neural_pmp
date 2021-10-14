# File containing common types of networks. More specialized networks are in other files.

import torch
import torch.nn as nn
#import torch.nn.functional as F

class Mlp(nn.Module):
    """
    Simple multi-layer perceptron net (densly connected net)
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        layer_dims (List[int]): Dimensions of hidden layers
        activation (str): type of activations. Not applying to the last layer 
    """
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
                elif activation == 'relu':
                    self.layers.append(nn.ReLU())
                self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'relu':
                    self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(layer_dims[-1], output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, output_dim))
        # Composing all layers
        self.net = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.net(x)

class CNNEncoder(nn.Module):
    """
    A CNN encoder that map 3D tensor to 1D tensor (1D can be compressed or full info)
    Each time image size is cut by half
    Args:
        channels (List[int]): list of number of channels to be applied starting with original number of channels of 3D tensor input
        activation (str): type of activations. 
    """
    def __init__(self, channels, activation='tanh'):
        super(CNNEncoder, self).__init__()
        self.layers = []
        for i in range(len(channels)-1):
            self.layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.BatchNorm2d(channels[i+1]))
            if activation == 'tanh':
                    self.layers.append(nn.Tanh())
            elif activation == 'relu':
                self.layers.append(nn.ReLU())
        # Composing all convolutional layers
        self.net = nn.Sequential(*self.layers, nn.Flatten())
    
    def forward(self, x):
        return self.net(x)
    
# Opposite to encoder
class CNNDecoder(nn.Module):
    """
    A CNN decoder that map 1D tensor to 3D tensor (1D can be compressed or full info)
    Each time image size is cut by half
    Args:
        channels (List[int]): list of number of channels to be applied starting with original number of channels of 1D tensor
        img_dim (int): width-height dimension of 3D tensor
        activation (str): type of activations. 
    """
    def __init__(self, channels, img_dim, activation='tanh'):
        super(CNNDecoder, self).__init__()
        self.layers = []
        for i in range(len(channels)-1):
            self.layers.append(nn.ConvTranspose2d(channels[i], channels[i+1],\
                                        kernel_size=3, stride=2, padding=1, output_padding=1))
            self.layers.append(nn.BatchNorm2d(channels[i+1]))
            if activation == 'tanh':
                    self.layers.append(nn.Tanh())
            elif activation == 'relu':
                self.layers.append(nn.ReLU())
        # Composing all convolutional layers
        width_outdim = img_dim//(2**(len(channels)-1))
        self.net = nn.Sequential(nn.Unflatten(1, 
                torch.Size([channels[0], width_outdim, width_outdim])), *self.layers)
    
    def forward(self, x):
        return self.net(x)
    
class Encoder(nn.Module):
    """
    A multilayer perceptron encoder that map 1D tensor to two mean and (diagonal) log-variance 1D tensors 
    Args:
        channels (List[int]): list of number of channels to be applied starting with original number of channels of 3D tensor input
        activation (str): type of activations. 
    """
    def __init__(self, input_dim, output_dim, share_layer_dims=[], 
                 mean_layer_dims=[], logvar_layer_dims=[], activation='tanh'):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        # intermediate dimension
        idim = share_layer_dims[-1]
        self.share_mlp = Mlp(input_dim=input_dim, layer_dims=share_layer_dims[:-1], output_dim=idim)
        self.mean_mlp = Mlp(input_dim=idim, layer_dims=mean_layer_dims, output_dim=output_dim)
        self.logvar_mlp = Mlp(input_dim=idim, layer_dims=logvar_layer_dims, output_dim=output_dim)
    
    def forward(self, x):
        # common output head
        com_head = self.share_mlp(x)
        if self.activation == 'tanh':
            com_head = torch.tanh(com_head)
        elif self.activation == 'relu':
            com_head = torch.relu(com_head)
        mu = self.mean_mlp(com_head)
        logvar = self.logvar_mlp(com_head)
        
        return mu, logvar