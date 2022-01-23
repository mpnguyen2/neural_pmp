import argparse
import numpy as np
import torch
import pandas as pd

from common.common_nets import Mlp, Encoder

#from envs.classical_controls import MountainCar, Pendulum, CartPole
#from envs.density_optimization import DensityOpt

from train_utils import training, get_environment, get_architectures, get_train_params

device_default = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_env(env_name, stochastic=False, sigma=1, device=device_default, 
        activation='tanh', load_model=False, num_train=10, num_warmup=10,
        num_episodes_hnet=1024, num_episodes_adj=2048, rate=1.5, 
        num_hnet_train_max=10000, num_adj_train_max=1000,
        batch_size_hnet=32, batch_size_hnet_sample=256, batch_size_adj=64, 
        update_interval_custom=-1, log_interval_custom=-1, stop_train_condition=0.01, 
        model_dir = 'models/', lr_change=False, lr_hnet_custom=1e-3, lr_adj_custom=1e-3):

    # File names for architectures and hyperparams
    arch_file=model_dir + 'architectures.csv'; param_file=model_dir + 'train_params.csv'
    # Initialize networks with specific architectures detailed in arch_file
    _, adj_net, hnet, hnet_target = get_architectures(arch_file, env_name, activation)
    # Initialize hyperparameters detailed in param_file
    T_hnet, T_adj, n_timesteps, control_coef, lr_hnet, lr_adj, update_interval, log_interval_hnet = get_train_params(param_file, env_name)
    # Get environment (with specific (quadratic) control coefficient)
    env = get_environment(env_name, control_coef=control_coef)
    # Set custom hyperparameter if allowed
    if update_interval_custom != -1:
        update_interval = update_interval_custom
    if log_interval_custom != -1:
        log_interval_hnet = log_interval_custom
    lr_hnet = lr_hnet_custom if lr_change else lr_hnet
    lr_adj = lr_adj_custom if lr_change else lr_adj
    
    #Print hyperparameters info
    print(f'\nDevice: {device}')
    print('\nHamiltonian training:')
    print(f'total number of episodes: {num_episodes_hnet}, max number of iterations: {num_hnet_train_max}.')
    print(f'update interval: {update_interval}, rate to train while sampling: {rate}.')
    print(f'sample_batch_size: {batch_size_hnet_sample}, batch size: {batch_size_hnet}, learning_rate: {lr_hnet}.')
    print(f'\nAdjoint net training:')
    print(f'total number of episodes: {num_episodes_adj}, max number of iterations: {num_adj_train_max}')
    print(f'batch size: {batch_size_adj}, learning rate: {lr_adj}')
    if stochastic:
        print(f'\nThe sigma constant for the diffusion is {sigma}.')

    # Training step
    version = 'stochastic' if stochastic else 'deterministic'
    print(f'\nTraining environment {env_name} (' + version + ' version):')

    log_interval_adj = 100
    training(stochastic, sigma, device, env, env_name, 
        adj_net, hnet, hnet_target, 
        num_train, num_warmup, load_model,
        T_hnet, T_adj, n_timesteps, control_coef,
        num_episodes_hnet, num_episodes_adj, 
        update_interval, rate, 
        batch_size_hnet_sample, batch_size_hnet, batch_size_adj,  
        lr_hnet, lr_adj, log_interval_hnet, log_interval_adj,
        num_hnet_train_max, num_adj_train_max, stop_train_condition)

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='CLI argument for testing')
    # Most general info
    parser.add_argument('env_name', help='Environment to train neural pmp on: mountain car|pendulum|cartpole|shape_opt|shape_opt_boundary')
    parser.add_argument('--stochastic', type=bool, default=False, help='Whether to train the stochastic version or not')
    parser.add_argument('--sigma', type=float, default=0.1, help='sigma factor of the diffusion')
    parser.add_argument('--device', default=device_default, help='Device for models to be trained on: GPU, CPU, etc')
    parser.add_argument('--num_train', type=int, default=10, help='How many phases to train after warmup step')
    parser.add_argument('--num_warmup', type=int, default=10, help='How many phases in warmup training')
    parser.add_argument('--load_model', type=bool, default=False, help='Whether to load network weights from files before training')
    parser.add_argument('--activation', type=str, default='tanh', help='Activation type for network')
    # Number of episode and iterations
    parser.add_argument('--num_episodes_hnet', type=int, default=512, help='Number of total trajectories to be sampled during hamiltonian net training')
    parser.add_argument('--num_episodes_adj', type=int, default=256, help='Number of total trajectories to be sampled during adjoint net training')
    parser.add_argument('--num_hnet_train_max', type=int, default=100000, help='Number of maximum (additional) training steps for reduced Hamiltonian')
    parser.add_argument('--num_adj_train_max', type=int, default=1000, help='Number of maximum training steps for adjoint network')
    # Number of batch size and learning rate
    parser.add_argument('--batch_size_hnet', type=int, default=32, help='Batch size for hamiltonian network training')
    parser.add_argument('--batch_size_hnet_sample', type=int, default=32, help='Batch size for hnet sampling training')
    parser.add_argument('--batch_size_adj', type=int, default=128, help='Batch size for adjacent network training')
    parser.add_argument('--lr_change', type=bool, default=False, help='Whether to change learning rates by yourself')
    parser.add_argument('--lr_hnet', type=float, default=1e-3, help='Learning rate for hamiltonian network')
    parser.add_argument('--lr_adj', type=float, default=1e-3, help='Learning rate for adjoint network')
    # Specific hyperparameter for reduced hamiltonian training
    parser.add_argument('--rate', type=float, default=1, help='Training rate everything we get a sample from the trajectory')
    parser.add_argument('--update_interval', type=int, default=-1, help='number of times to optimize Hamiltonian net before sampling')
    parser.add_argument('--log_interval', type=int, default=-1, help='Training log interval')
    parser.add_argument('--stop_train_condition', type=float, default=0.01, help='Condition when to stop the training (early)')

    # Call train environment
    args = parser.parse_args()
    train_env(env_name=args.env_name, stochastic=args.stochastic, sigma=args.sigma, device=args.device, 
        num_train=args.num_train, num_warmup=args.num_warmup, load_model=args.load_model, activation=args.activation,
        num_episodes_hnet=args.num_episodes_hnet, num_episodes_adj=args.num_episodes_adj, rate=args.rate, 
        num_hnet_train_max=args.num_hnet_train_max, num_adj_train_max=args.num_adj_train_max,
        batch_size_hnet=args.batch_size_hnet, batch_size_hnet_sample=args.batch_size_hnet_sample, batch_size_adj=args.batch_size_adj,   
        update_interval_custom=args.update_interval, log_interval_custom=args.log_interval, stop_train_condition=args.stop_train_condition, 
        lr_change=args.lr_change, lr_hnet_custom=args.lr_hnet, lr_adj_custom=args.lr_adj)