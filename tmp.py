import torch
from train_utils import train_phase_2
from common.common_nets import Mlp, Encoder
from envs.classical_controls import Pendulum

# 5 10
a = torch.tensor([[1, 1, 2], [3, 4, 5]])
b = torch.tensor([[1, 0, 2], [2, 1, 0]])

c = torch.einsum('ij, ij->i', a, b)
#print(c.shape)

num_examples = 500

# Previously stored architectures
q_dim=2; z_dim=2
adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
model_name='pendulum'
# New architectures
HnetDecoder = Mlp(input_dim=2*q_dim, output_dim=2*q_dim, layer_dims=[8, 16, 32])
# qpt to and from z
z_encoder = Encoder(input_dim=2*q_dim, share_layer_dims=[32], 
                 mean_layer_dims=[16, 4], logvar_layer_dims=[16, 4], output_dim=z_dim)
z_decoder = Mlp(input_dim=z_dim, output_dim=2*q_dim, layer_dims = [16, 64])
env = Pendulum()
qs = torch.tensor(env.sample_q(num_examples), dtype=torch.float)

train_phase_2(adj_net, h_net, HnetDecoder, z_decoder, z_encoder, qs, 
                  T2 = 1.0, beta=1.0, 
                  batch_size=10, num_epoch=2, lr=1e-3, 
                  log_interval=5, model_name=model_name)