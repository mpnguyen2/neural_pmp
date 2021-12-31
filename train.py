import argparse
import numpy as np
import torch
import pandas as pd

from common.common_nets import Mlp, Encoder

#from envs.classical_controls import MountainCar, Pendulum, CartPole
#from envs.density_optimization import DensityOpt

from train_utils import training, get_environment, get_architectures, get_train_params

device_default = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_env(env_name, stochastic=False, sigma=1, device=device_default, num_episodes=20, rate=1.5, batch_size=32, 
        num_hnet_train_max=10000, num_adjoint_train_max=1000, stop_train_condition=0.01, retrain_phase1=True, 
        model_dir = 'models/', lr_change=False, lr_hnet_custom=1e-3, lr_adj_custom=1e-3):
    # Get and print hyperparameters info
    print(f'\nGeneral info for training hyperparameters:')
    print(f'Device: {device}, number of total trajectories used: {num_episodes}, rate at which to train hamiltonian while sampling: {rate}.')
    print(f'Batch size: {batch_size}, number of maximum training iteration for Hamiltonian'
        + f' is {num_hnet_train_max} and for Adjoint network is {num_adjoint_train_max}.')
    if lr_change:
        print(f'Learning rate for hamiltonian net is {lr_hnet_custom}, and for adjoint net is {lr_adj_custom}.')
    if stochastic:
        print(f'The sigma constant for the diffusion is {sigma}.')
    env = get_environment(env_name)
    arch_file=model_dir + 'architectures.csv'; param_file=model_dir + 'train_params.csv'
    # Initialize networks with specific architectures detailed in arch_file
    _, adj_net, hnet, hnet_target = get_architectures(arch_file, env_name)
    # Initialize hyperparameters detailed in param_file
    T, n_timesteps, control_coef, lr_hnet, lr_adj, update_interval, log_interval = get_train_params(param_file, env_name)
    if not lr_change:
        print(f'Learning rate for hamiltonian net is {lr_hnet}, and for adjoint net is {lr_adj}.')

    # Training step
    version = 'stochastic' if stochastic else 'deterministic'
    print(f'\nTraining environment {env_name} (' + version + ' version):')
    training(stochastic, sigma, device, env, env_name, num_episodes,
        adj_net, hnet, hnet_target,
        T, n_timesteps, control_coef, batch_size, update_interval, rate,
        num_hnet_train_max, num_adjoint_train_max, stop_train_condition,
        lr_hnet, lr_adj, log_interval, retrain_phase1)

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='CLI argument for testing')
    parser.add_argument('env_name', help='Environment to train neural pmp on: mountain car|pendulum|cartpole|shape_opt|shape_opt_boundary')
    parser.add_argument('--stochastic', type=bool, default=False, help='Whether to train the stochastic version or not')
    parser.add_argument('--sigma', type=float, default=0.1, help='sigma factor of the diffusion')
    parser.add_argument('--device', default=device_default, help='Device for models to be trained on: GPU, CPU, etc')
    parser.add_argument('--num_episodes', type=int, default=20, help='Number of total trajectories to be sampled during training')
    parser.add_argument('--rate', type=float, default=1.5, help='Training rate everything we get a sample from the trajectory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_hnet_train_max', type=int, default=50000, help='Number of maximum (additional) training steps for reduced Hamiltonian')
    parser.add_argument('--num_adjoint_train_max', type=int, default=5000, help='Number of maximum training steps for adjoint network')
    parser.add_argument('--stop_train_condition', type=float, default=0.001, help='Condition when to stop the training (early)')
    parser.add_argument('--lr_change', type=bool, default=False, help='Whether to change learning rates by yourself')
    parser.add_argument('--lr_hnet_custom', type=float, default=1e-3, help='Learning rate for hamiltonian network')
    parser.add_argument('--lr_adj_custom', type=float, default=1e-3, help='Learning rate for adjoint network')
    #parser.add_argument('num_episodes', help='Number of trajectories to be trained')
    args = parser.parse_args()
    # Call train environment
    train_env(env_name=args.env_name, stochastic=args.stochastic, sigma=args.sigma, device=args.device, num_episodes=args.num_episodes, 
        rate=args.rate, batch_size=args.batch_size, num_hnet_train_max=args.num_hnet_train_max, num_adjoint_train_max=args.num_adjoint_train_max,
        stop_train_condition=args.stop_train_condition, lr_change=args.lr_change, 
        lr_hnet_custom=args.lr_hnet_custom, lr_adj_custom=args.lr_adj_custom)