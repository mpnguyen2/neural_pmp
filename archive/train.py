import numpy as np
import torch
import wandb

from common.common_nets import Mlp, Encoder

from envs.classical_controls import MountainCar, Pendulum, CartPole
from envs.density_optimization import DensityOpt

from train_utils import train_phase_1, train_phase_2

# Wandb Tracking
wandb.init(project="NeuralPMP")

## Phase 1 training ##
train_mt, train_cart, train_pendulum, train_density = False, False, True, False

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
                  T1 = 1.0, dynamic_hidden=False, 
                  alpha1=1, alpha2=0.1, beta=1, 
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
                  T1=1.0, dynamic_hidden=False,
                  alpha1=1, alpha2=0.1, beta=1,
                  batch_size=10, num_epoch=2, lr = 1e-3, 
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
                  T1 = 1.0, control_coef = 1, dynamic_hidden=False,
                  alpha1=1, alpha2=0.1, beta=1,
                  batch_size=10, num_epoch=10, lr = 1e-3, 
                  log_interval=400, model_name='pendulum')
    
if train_density:
    #DEBUG: Shape optimization
    num_examples = 2000
    q_dim = 16; u_dim = 16
    env = DensityOpt()
    # q samples
    q_samples = torch.tensor(env.sample_q(num_examples, mode='train'), dtype=torch.float)
    
    # Net architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[32, 64])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[64, 8])
    
    print('\nTraining shape optimization...')
    train_phase_1(env, adj_net, h_net, qs=q_samples, 
                  T1=1.0, dynamic_hidden=False,
                  alpha1=0, alpha2=1, beta=1,
                  batch_size=10, num_epoch=2, lr = 1e-3, 
                  log_interval=40, model_name='shape_opt')
    
## Phase 2 training ##
train_mt2, train_cart2, train_pendulum2, train_density2 = False, False, False, False

if train_mt2:
    print('\nTraining mountain car phase 2...')
    num_examples = 10000
    q_dim = 2; z_dim = 2
    model_name='mountain_car'
    # Phase 1 architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
    # Phase 2 architectures: decoder reverse hamiltonian and encoder/decoder to/from latent
    HnetDecoder = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
    z_encoder = Encoder(input_dim=2*q_dim, share_layer_dims=[32], 
                     mean_layer_dims=[8], logvar_layer_dims=[8], output_dim=z_dim)
    z_decoder = Mlp(input_dim=z_dim, output_dim=2*q_dim, layer_dims = [8, 32])
    # Create env and train phase 2
    env = MountainCar()
    qs = torch.tensor(env.sample_q(num_examples), dtype=torch.float)
    train_phase_2(adj_net, h_net, HnetDecoder, z_decoder, z_encoder, qs, 
                      T2 = 1.0, beta=1.0, 
                      batch_size=10, num_epoch=10, lr=1e-3, 
                      log_interval=40, model_name=model_name)
        
if train_cart2:
    print('\nTraining cartpole phase 2...')
    num_examples = 10000
    q_dim = 4; z_dim = 4
    model_name='cartpole'
    # Phase 1 architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[16, 32, 32])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[16, 32, 64, 8])
    # Phase 2 architectures: decoder reverse hamiltonian and encoder/decoder to/from latent
    HnetDecoder = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=[16, 32, 64, 8])
    z_encoder = Encoder(input_dim=2*q_dim, share_layer_dims=[64], 
                     mean_layer_dims=[16], logvar_layer_dims=[16], output_dim=z_dim)
    z_decoder = Mlp(input_dim=z_dim, output_dim=2*q_dim, layer_dims = [16, 64])
    # Create env and train phase 2
    env = CartPole()
    qs = torch.tensor(env.sample_q(num_examples), dtype=torch.float)
    train_phase_2(adj_net, h_net, HnetDecoder, z_decoder, z_encoder, qs, 
                      T2 = 1.0, beta=1.0, 
                      batch_size=10, num_epoch=2, lr=1e-3, 
                      log_interval=40, model_name=model_name)
        
if train_pendulum2:
    print('\nTraining pendulum phase 2...')
    num_examples = 2000
    q_dim = 2; z_dim = 2
    model_name='pendulum'
    # Phase 1 architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
    # Phase 2 architectures: decoder reverse hamiltonian and encoder/decoder to/from latent
    HnetDecoder = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
    z_encoder = Encoder(input_dim=2*q_dim, share_layer_dims=[32], 
                     mean_layer_dims=[8], logvar_layer_dims=[8], output_dim=z_dim)
    z_decoder = Mlp(input_dim=z_dim, output_dim=2*q_dim, layer_dims = [8, 32])
    # Create env and train phase 2
    env = Pendulum()
    qs = torch.tensor(env.sample_q(num_examples), dtype=torch.float)
    train_phase_2(adj_net, h_net, HnetDecoder, z_decoder, z_encoder, qs, 
                      T2 = 1.0, beta=1.0, 
                      batch_size=10, num_epoch=2, lr=1e-3, 
                      log_interval=40, model_name=model_name)

if train_density2:
    print('\nTraining shape optimization phase 2...')
    num_examples = 10000
    q_dim = 16; z_dim=4
    model_name='shape_opt'
    # Phase 1 architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[32, 64])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[64, 8])
    # Phase 2 architectures: decoder reverse hamiltonian and encoder/decoder to/from latent
    HnetDecoder = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=[64, 8])
    # qpt to and from z
    z_encoder = Encoder(input_dim=2*q_dim, share_layer_dims=[64], 
                     mean_layer_dims=[16, 8], logvar_layer_dims=[16, 8], output_dim=z_dim)
    z_decoder = Mlp(input_dim=z_dim, output_dim=2*q_dim, layer_dims = [8, 16, 64])
    
    # Create env and train phase 2
    env = DensityOpt()
    qs = torch.tensor(env.sample_q(num_examples), dtype=torch.float)
    
    train_phase_2(adj_net, h_net, HnetDecoder, z_decoder, z_encoder, qs, 
                      T2 = 1.0, beta=1.0, 
                      batch_size=10, num_epoch=2, lr=1e-3, 
                      log_interval=40, model_name=model_name)
