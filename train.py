import argparse
import numpy as np
import torch
import pandas as pd

from common.common_nets import Mlp, Encoder

#from envs.classical_controls import MountainCar, Pendulum, CartPole
#from envs.density_optimization import DensityOpt

from train_utils import training, get_environment, get_architectures, get_train_params

device_default = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_env(env_name, stochastic=False, sigma=1, device=device_default, retrain_hnet=True,
        num_episodes_hnet=1024, num_episodes_adj=2048, rate=1.5, 
        num_hnet_train_max=10000, num_adj_train_max=1000,
        batch_size_hnet=32, batch_size_hnet_sample=256, batch_size_adj=64, 
        update_interval_custom=-1, log_interval_custom=-1, stop_train_condition=0.01, 
        model_dir = 'models/', lr_change=False, lr_hnet_custom=1e-3, lr_adj_custom=1e-3):
    # Get and print hyperparameters info
    print(f'\nGeneral info for training hyperparameters:')
    print(f'Device: {device}')
    print(f'Hamiltonian training: sample_batch_size {batch_size_hnet_sample}, batch size {batch_size_hnet}, rate to train while sampling: {rate}.')
    print(f'Hamiltonian training: total number of episodes used {num_episodes_hnet}, number of maximum training iteration for Hamiltonian is {num_hnet_train_max}.')
    print(f'Adjoint network training: total number of episodes used {num_episodes_adj}, batch size {batch_size_adj}.')
    print(f'Adjont network training: number of maximum training iteration for adjoint network is {num_adj_train_max}')
    if lr_change:
        print(f'Learning rate for hamiltonian net is {lr_hnet_custom}, and for adjoint net is {lr_adj_custom}.')
    if stochastic:
        print(f'The sigma constant for the diffusion is {sigma}.')
    # File names for architectures and hyperparams
    arch_file=model_dir + 'architectures.csv'; param_file=model_dir + 'train_params.csv'
    # Initialize networks with specific architectures detailed in arch_file
    _, adj_net, hnet, hnet_target = get_architectures(arch_file, env_name)
    # Initialize hyperparameters detailed in param_file
    T_hnet, T_adj, n_timesteps, control_coef, lr_hnet, lr_adj, update_interval, log_interval_hnet = get_train_params(param_file, env_name)
    # Get environment (with specific (quadratic) control coefficient)
    env = get_environment(env_name, control_coef=control_coef)
    # Set custom hyperparameter if allowed
    if not lr_change:
        print(f'Learning rate for hamiltonian net is {lr_hnet}, and for adjoint net is {lr_adj}.')
    if update_interval_custom != -1:
        update_interval = update_interval_custom
    if log_interval_custom != -1:
        log_interval_hnet = log_interval_custom
    print(f'Update interval (# times to minimize Hamiltonian before sampling): {update_interval}')

    # Training step
    version = 'stochastic' if stochastic else 'deterministic'
    print(f'\nTraining environment {env_name} (' + version + ' version):')

    training(stochastic, sigma, device, env, env_name, adj_net, hnet, hnet_target,
        num_episodes_hnet, T_hnet, n_timesteps, control_coef,
        batch_size_hnet_sample, batch_size_hnet, update_interval, rate, lr_hnet, 
        num_hnet_train_max, log_interval_hnet, stop_train_condition, retrain_hnet, 
        num_episodes_adj, num_adj_train_max, T_adj, batch_size_adj, lr_adj, log_interval_adj=1)

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='CLI argument for testing')
    # Most general info
    parser.add_argument('env_name', help='Environment to train neural pmp on: mountain car|pendulum|cartpole|shape_opt|shape_opt_boundary')
    parser.add_argument('--stochastic', type=bool, default=False, help='Whether to train the stochastic version or not')
    parser.add_argument('--sigma', type=float, default=0.1, help='sigma factor of the diffusion')
    parser.add_argument('--device', default=device_default, help='Device for models to be trained on: GPU, CPU, etc')
    parser.add_argument('--retrain_hnet', type=bool, default=False, help='Device for models to be trained on: GPU, CPU, etc')
    # Number of episode and iterations
    parser.add_argument('--num_episodes_hnet', type=int, default=1024, help='Number of total trajectories to be sampled during hamiltonian net training')
    parser.add_argument('--num_episodes_adj', type=int, default=2048, help='Number of total trajectories to be sampled during adjoint net training')
    parser.add_argument('--num_hnet_train_max', type=int, default=1000000, help='Number of maximum (additional) training steps for reduced Hamiltonian')
    parser.add_argument('--num_adj_train_max', type=int, default=1000, help='Number of maximum training steps for adjoint network')
    # Number of batch size and learning rate
    parser.add_argument('--batch_size_hnet', type=int, default=64, help='Batch size for hamiltonian network training')
    parser.add_argument('--batch_size_hnet_sample', type=int, default=256, help='Batch size for hnet sampling training')
    parser.add_argument('--batch_size_adj', type=int, default=64, help='Batch size for adjacent network training')
    parser.add_argument('--lr_change', type=bool, default=False, help='Whether to change learning rates by yourself')
    parser.add_argument('--lr_hnet', type=float, default=1e-3, help='Learning rate for hamiltonian network')
    parser.add_argument('--lr_adj', type=float, default=1e-3, help='Learning rate for adjoint network')
    # Specific hyperparameter for reduced hamiltonian training
    parser.add_argument('--rate', type=float, default=1.5, help='Training rate everything we get a sample from the trajectory')
    parser.add_argument('--update_interval', type=int, default=-1, help='number of times to optimize Hamiltonian net before sampling')
    parser.add_argument('--log_interval', type=int, default=-1, help='Training log interval')
    parser.add_argument('--stop_train_condition', type=float, default=0.001, help='Condition when to stop the training (early)')

    # Call train environment
    args = parser.parse_args()
    train_env(env_name=args.env_name, stochastic=args.stochastic, sigma=args.sigma, device=args.device, retrain_hnet=args.retrain_hnet,
        num_episodes_hnet=args.num_episodes_hnet, num_episodes_adj=args.num_episodes_adj, rate=args.rate, 
        num_hnet_train_max=args.num_hnet_train_max, num_adj_train_max=args.num_adj_train_max,
        batch_size_hnet=args.batch_size_hnet, batch_size_hnet_sample=args.batch_size_hnet_sample, batch_size_adj=args.batch_size_adj,   
        update_interval_custom=args.update_interval, log_interval_custom=args.log_interval, stop_train_condition=args.stop_train_condition, 
        lr_change=args.lr_change, lr_hnet_custom=args.lr_hnet, lr_adj_custom=args.lr_adj)