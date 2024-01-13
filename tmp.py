import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint

from train_utils import train_phase_2
from common.common_nets import Mlp, Encoder
from model_nets import HDNet
from envs.classical_controls import Pendulum


# 5 10
a = torch.tensor([[1, 1, 2], [3, 4, 5]])
b = torch.tensor([[1, 0, 2], [2, 1, 0]])

c = torch.einsum('ij, ij->i', a, b)
#print(c.shape)
'''
env = Pendulum()

# models
q_dim=2; 
adj_net= Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
Hnet = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
# Build symplectic dynamics net from Hamiltonian net
HDnet = HDNet(Hnet=Hnet)

# Run optimal trajectory
#q = torch.tensor(env.sample_q(1, mode='test'), dtype=torch.float)
q = torch.zeros(1, 2)
p = adj_net(q)
qp = torch.cat((q, p), axis=1)
time_steps = list(np.arange(0, 10, 0.01))
traj = odeint(HDnet, qp, torch.tensor(time_steps, requires_grad=False))
print('Done finding trajectory...')
# Print trajectory and save images to vid
cnt = 0
for e in traj:
    qe, _ = torch.chunk(e, 2, dim=1)
    qe_np = qe.detach().numpy()
    env.render(qe_np.reshape(-1))
    cnt += 1

env.close()
'''

'''
q_dim=2; model_name='mountain_car'
adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
adj_net.load_state_dict(torch.load('models/' + model_name + '/adjoint.pth'))
h_net.load_state_dict(torch.load('models/' + model_name + '/hamiltonian_dynamics.pth'))
print('Adjoint net...')
for param in adj_net.parameters():
    print(param.data)

print('H net...')
for param in h_net.parameters():
    print(param.data)
'''