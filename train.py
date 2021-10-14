import numpy as np
import torch

from common.common_nets import Mlp

from envs.classical_controls import MountainCar, Pendulum, CartPole
from envs.density_optimization import DensityOpt

from train_utils import train_phase_1

train_mt, train_cart, train_pendulum, train_density = True, False, False, False

if train_mt:
    #DEBUG: Mountain Car
    num_examples = 20000
    q_dim = 2; u_dim = 1
    env = MountainCar()
    # q samples
    q_samples = torch.tensor(env.sample_q(num_examples, mode='train'), dtype=torch.float)
    
    # Net architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
    
    print('\nTraining mountain car...')
    train_phase_1(env, adj_net, h_net, qs=q_samples, 
                  T1 = 1.0, dynamic_hidden=False, alpha=1,
                  batch_size=10, num_epoch=2, lr = 1e-3, 
                  log_interval=400, model_name='mountain_car')

if train_cart:
    #DEBUG: Cartpole
    num_examples = 20000
    q_dim = 4; u_dim = 1
    env = CartPole()
    # q samples
    q_samples = torch.tensor(env.sample_q(num_examples, mode='train'), dtype=torch.float)
    
    # Net architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[16, 32, 32])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[16, 32, 64, 8])
    
    print('\nTraining cartpole...')
    train_phase_1(env, adj_net, h_net, qs=q_samples, 
                  dynamic_hidden=False, alpha=1,
                  batch_size=10, num_epoch=5, lr = 1e-3, 
                  log_interval=400, model_name='cartpole')

if train_pendulum:
    #DEBUG: Pendulum
    num_examples = 40000
    q_dim = 2; u_dim = 1
    env = Pendulum()
    # q samples
    q_samples = torch.tensor(env.sample_q(num_examples, mode='train'), dtype=torch.float)
    
    # Net architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
    
    print('\nTraining pendulum...')
    train_phase_1(env, adj_net, h_net, qs=q_samples, 
                  T1 = 1.0, scale = 1, dynamic_hidden=False, alpha=1,
                  batch_size=10, num_epoch=5, lr = 1e-3, 
                  log_interval=400, model_name='pendulum')
    
if train_density:
    #DEBUG: Shape optimization
    num_examples = 2000
    q_dim = 16; u_dim = 16
    env = DensityOpt()
    # q samples
    q_samples = torch.tensor(env.sample_q(num_examples, mode='train'), dtype=torch.float)
    
    # Net architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[32, 64, 128])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[64, 128, 32, 8])
    
    print('\nTraining shape optimization...')
    train_phase_1(env, adj_net, h_net, qs=q_samples, 
                  T1=1.0, dynamic_hidden=False, alpha=1,
                  batch_size=10, num_epoch=2, lr = 1e-3, 
                  log_interval=40, model_name='shape_opt')