import numpy as np
import torch
import pandas as pd

from common.common_nets import Mlp, Encoder

#from envs.classical_controls import MountainCar, Pendulum, CartPole
#from envs.density_optimization import DensityOpt

from train_utils import training, get_environment, get_architectures, get_train_params
        
def train_env(env_name, num_examples, mode=0, 
              retrain_phase1=True, retrain_phase2=True,
              num_additional_train=5, num_examples_phase2=0.2,
              arch_file='models/architectures.csv', param_file='models/train_params.csv'):
    print(f'\nTraining environment {env_name}:\n')
    env = get_environment(env_name)
    _, adj_net, hnet, hnet_decoder, z_encoder, z_decoder = get_architectures(arch_file, env_name)
    T1, T2, control_coef, dynamic_hidden, alpha1, alpha2, beta1, beta2, \
        batch_size1, num_epoch1, lr1, log_interval1,\
        batch_size2, num_epoch2, lr2, log_interval2 = get_train_params(param_file, env_name)
    T1, T2 = float(T1), float(T2)
    q_samples = torch.tensor(env.sample_q(num_examples, mode='train'), dtype=torch.float)
    training(env, env_name, q_samples, 
        adj_net, hnet, hnet_decoder, z_decoder, z_encoder, 
        T1=T1, T2=T2, control_coef=control_coef, dynamic_hidden=dynamic_hidden,
        alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2,
        batch_size1=batch_size1, num_epoch1=num_epoch1, lr1=lr1, log_interval1=log_interval1, 
        batch_size2=batch_size2, num_epoch2=num_epoch2, lr2=lr2, log_interval2=log_interval2,
        mode=mode, retrain_phase1=retrain_phase1, retrain_phase2=retrain_phase2,
        num_examples_phase2=num_examples_phase2, num_additional_train=num_additional_train)


train_mt, train_cart, train_pendulum, train_density = True, False, False, False

if train_mt:
    train_env('mountain_car', num_examples=2000, mode=1, num_examples_phase2=1, retrain_phase1=True)
