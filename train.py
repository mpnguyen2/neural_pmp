import argparse
import numpy as np
import torch
import pandas as pd

from common.common_nets import Mlp, Encoder

#from envs.classical_controls import MountainCar, Pendulum, CartPole
#from envs.density_optimization import DensityOpt

from train_utils import training, get_environment, get_architectures, get_train_params

device_default = torch.device("cuda" if torch.cuda.is_available else "cpu")

def train_env(env_name, device=device_default, num_trajs=20, rate=2, num_hnet_train_max=10000, num_adjoint_train_max=1000, 
            stop_train_condition=0.001, retrain_phase1=True, model_dir = 'models/'):
    print(f'\nTraining environment {env_name}:\n')
    env = get_environment(env_name)
    arch_file=model_dir + 'architectures.csv'; param_file=model_dir + 'train_params.csv'
    # Initialize networks with specific architectures detailed in arch_file
    _, adj_net, hnet, hnet_target = get_architectures(arch_file, env_name)
    # Initialize hyperparameters detailed in param_file
    T, n_timesteps, control_coef, batch_size,\
        lr_hnet, lr_adj, update_interval, log_interval = get_train_params(param_file, env_name)

    # Training step
    training(device, env, env_name, num_trajs,
        adj_net, hnet, hnet_target,
        T, n_timesteps, control_coef, batch_size, update_interval, rate,
        num_hnet_train_max, num_adjoint_train_max, stop_train_condition,
        lr_hnet, lr_adj, log_interval, retrain_phase1)

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('env_name', help='Environment to train neural pmp on: mountain car|pendulum|cartpole|shape_opt|shape_opt_boundary')
    parser.add_argument('--device', default=device_default, help='Device for models to be trained on: GPU, CPU, etc')
    parser.add_argument('--num_trajs', type=int, default=100, help='Number of total trajectories to be sampled during training')
    parser.add_argument('--rate', type=float, default=2, help='Training rate everything we get a sample from the trajectory')
    parser.add_argument('--num_hnet_train_max', type=int, default=10000, help='Number of maximum (additional) training steps for reduced Hamiltonian')
    parser.add_argument('--num_adjoint_train_max', type=int, default=1000, help='Number of maximum training steps for adjoint network')
    parser.add_argument('--stop_train_condition', type=float, default=0.001, help='Condition when to stop the training (early)')
    #parser.add_argument('num_trajs', help='Number of trajectories to be trained')
    args = parser.parse_args()
    # Call train environment
    train_env(env_name=args.env_name, device=args.device, num_trajs=args.num_trajs, rate=args.rate,
            num_hnet_train_max=args.num_hnet_train_max, num_adjoint_train_max=args.num_adjoint_train_max,
            stop_train_condition=args.stop_train_condition)