# reduced hamiltonian (q(1), -nabla g(q(1))) -> (q(0), p(0)) -> h(q(0), p(0)) = L to minimize
# Use backward dynamics: f = (-h_p, h_q)

import torch
from common_nets import Mlp
from model_nets import HDNet
from utils import generate_coords
from torchdiffeq import odeint_adjoint as odeint


def train(h_layer_dims=[], num_batch=1000, log_interval=5):
    # HDnet calculate the Hamiltonian dynamics network given the Hamiltonian network Hnet
    Hnet = Mlp(input_dim=32, output_dim = 1, layer_dims=h_layer_dims) 
    HDnet = HDNet(Hnet=Hnet)
    # Optimizer for HDnet
    optim = torch.optim.Adam(HDnet.parameters(), lr=3e-5)
    # Go over each batch of data and optimize HDnet
    for i in range(num_batch):
        # q_p is the generalized coordinate
        qp_one = generate_coords()

        # Given the generalized coordinate, use parameterized network to obtain the (backward) dynamic (function)
        # together with ODE net to obtain the q_p at time 0 (or time 1 in backward setting)
        qp_zero = odeint(HDnet, qp_one, torch.tensor([1.0], requires_grad=True))
        # Train loss function is the Hamiltonian at the time 0 generalized coordinate
        H0 = Hnet(qp_zero).sum()
        H0.backward()
        optim.step(); optim.zero_grad()
        if (i+1)% log_interval == 0:
            print('Loss for {}th batch is: {}'.format(i+1, H0.item()))
    # save model
    torch.save(HDnet.state_dict(), 'models/hd.pth')
    
    
train(h_layer_dims=[64, 16, 32, 8, 16, 2])