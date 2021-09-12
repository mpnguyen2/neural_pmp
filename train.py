# reduced hamiltonian (q(1), -nabla g(q(1))) -> (q(0), p(0)) -> h(q(0), p(0)) = L to minimize
# Use backward dynamics: f = (-h_p, h_q)
import numpy as np

import torch
from torchdiffeq import odeint_adjoint as odeint

from common.common_nets import Mlp
from common.utils import generate_coords
from model_nets import HDNet

def train(h_layer_dims=[], num_samples=1024, batch_size=32, num_epoch=20, 
          lr=1e-3, log_interval=50):
    # HDnet calculate the Hamiltonian dynamics network given the Hamiltonian network Hnet
    Hnet = Mlp(input_dim=32, output_dim = 1, layer_dims=h_layer_dims) 
    HDnet = HDNet(Hnet=Hnet)
    # Optimizer for HDnet
    optim = torch.optim.Adam(Hnet.parameters(), lr=lr)
    optim.zero_grad()
    # Go over each batch of data and optimize HDnet
    times = [0.0, 1.0]
    # Generate num_samples data q_p, the starting generalized coordinates
    X = generate_coords(num_samples=num_samples)
    for i in range(num_epoch):
        loss = 0
        Xcur = torch.clone(X)[torch.randperm(num_samples)]
        num_iter = num_samples//batch_size
        for j in range(num_iter):
            qp_one = Xcur[j*batch_size:(j+1)*batch_size]
            # Given the generalized coordinate, use parameterized network to obtain the (backward) dynamic (function)
            # together with ODE net to obtain the q_p at time 0 (or time 1 in backward setting)
            qp_zero = odeint(HDnet, qp_one, torch.tensor(times, requires_grad=True))[-1]
            # Train loss function is the Hamiltonian at the time 0 generalized coordinate
            dmpdq_zero= HDnet(0, qp_zero)
            dp_zero, dq_zero = torch.chunk(dmpdq_zero, 2, dim=1); dp_zero = -dp_zero
            _, p_zero = torch.chunk(qp_zero, 2, dim=1)
            H0 = torch.sum((dp_zero-p_zero)**2)
            H0.backward()
            optim.step(); optim.zero_grad()
            loss += H0.item()
        if (i+1)% log_interval == 0:
            print('Average loss for {}th epoch is: {}'.format(i+1, loss/(num_iter*batch_size)))
    # save model
    torch.save(Hnet.state_dict(), 'models/hd.pth')
    
# Train model  
train(h_layer_dims=[64, 16, 32, 8, 16, 2], num_epoch=1000, lr=1e-3, log_interval=1)