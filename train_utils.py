from collections import namedtuple, deque
import random
import time
import wandb

import numpy as np
import pandas as pd

import torch
from torchdiffeq import odeint, odeint_adjoint
from torchsde import sdeint

from common.common_nets import Mlp, Encoder
from model_nets import HDNet, HDStochasticNet, HDVAE, HDInverseNet
from envs.classical_controls import MountainCar, CartPole, Pendulum, TestEnv
from envs.density_optimization import DensityOpt, DensityOptBoundary

# Constant clipping value
MAX_VAL = 10.0
# Least number of Hamiltonian training before considering threshold conditions
LEAST_NUM_TRAIN = 10

def toList(s):
    tokens = s[1:-1].split(", ")
    ans = []
    for token in tokens:
        ans.append(int(token))
    return ans

## Reading architecture and training parameters routines ##
# Get correct environment
def get_environment(env_name, control_coef=0.5):
    if env_name == 'mountain_car':
        return MountainCar(control_coef=control_coef)
    if env_name == 'cartpole':
        return CartPole(control_coef=control_coef)
    if env_name == 'pendulum':
        return Pendulum(control_coef=control_coef)
    if env_name == 'shape_opt':
        return DensityOpt(control_coef=control_coef)
    if env_name == 'shape_opt_boundary':
        return DensityOptBoundary(control_coef=control_coef)
    if env_name == 'test_env':
        return TestEnv(control_coef=control_coef)
    
# Get architecture
def get_architectures(arch_file, env_name, activation='tanh', phase2=False):
    r"""
    Gets the architecture info from arch_file provided. 

    Args:
        arch_file (str): Path to the architecture.csv file
        env_name (str): Environment to train NeuralPMP on. Options: mountain car, pendulum, cartpole, shape_opt, shape_opt_boundary
        activation (str ): Activation function
        phase2 (bool): Build hydrogen bond net decoder, encoder, and latent z decoder (phase 2 of training)
    """
    # Get architecture info from arch_file
    df = pd.read_csv(arch_file)
    info = df[df['env_name']==env_name]
    # Extract architecture info
    q_dim=info['q_dim'].values[0]
    adj_net_layer_dims = toList(info['adj_net_layer_dims'].values[0])
    hnet_layer_dims = toList(info['hnet_layer_dims'].values[0])
    if phase2:
        hnet_decoder_layer_dims = toList(info['hnet_decoder_layer_dims'].values[0])
        z_dim = int(info['z_dim'].values[0])
        z_encoder_share_layer_dims = toList(info['z_encoder_share_layer_dims'].values[0])
        z_encoder_mean_layer_dims = toList(info['z_encoder_mean_layer_dims'].values[0])
        z_encoder_logvar_layer_dims = toList(info['z_encoder_logvar_layer_dims'].values[0])
        z_decoder_layer_dims = toList(info['z_decoder_layer_dims'].values[0])
    # Build adjoint variable and Hamiltonian nets
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=adj_net_layer_dims, activation=activation)
    hnet = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=hnet_layer_dims, activation=activation)
    hnet_target = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=hnet_layer_dims, activation=activation)
    if phase2:
        # Build hnet_decoder
        hnet_decoder = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=hnet_decoder_layer_dims)
        # Build latent z_encoder
        z_encoder = Encoder(input_dim=2*q_dim, share_layer_dims=z_encoder_share_layer_dims, 
            mean_layer_dims=z_encoder_mean_layer_dims, 
            logvar_layer_dims=z_encoder_logvar_layer_dims, 
            output_dim=z_dim)
        # Build latent z_decoder
        z_decoder = Mlp(input_dim=z_dim, output_dim=2*q_dim, layer_dims=z_decoder_layer_dims)
    
    if not phase2:
        return q_dim, adj_net, hnet, hnet_target

    return q_dim, adj_net, hnet, hnet_target, hnet_decoder, z_encoder, z_decoder
    
def get_train_params(param_file, env_name):
    # Get parameter info from param file
    df = pd.read_csv(param_file)
    info = df[df['env_name']==env_name]
    # Get terminal times T
    T_hnet, T_adj, n_timesteps = info['T_hnet'].values[0], info['T_adj'].values[0], info['n_timesteps'].values[0]
    # Get control coefficients 
    control_coef = info['control_coef'].values[0]
    # Get training details for first phase (batch_size, num_epoch, lr, log_interval)
    # Currently refocus on first phase
    lr_hnet, lr_adj, update_interval, log_interval = info['lr_hnet'].values[0],\
        info['lr_adj'].values[0], info['update_interval'].values[0], info['log_interval'].values[0]

    return float(T_hnet), float(T_adj), int(n_timesteps), control_coef, lr_hnet, lr_adj, int(update_interval), int(log_interval)

def get_test_params(param_file, env_name):
    pass

## Saving models ##
# save model phase 1
def save_models(adj_net, hnet, env_name):
    torch.save(adj_net.state_dict(), 'models/' + env_name + '/adjoint.pth')
    torch.save(hnet.state_dict(), 'models/' + env_name + '/hamiltonian_dynamics.pth')

# load model phase 1
def load_models(adj_net, hnet, env_name):
    adj_net.load_state_dict(torch.load('models/' + env_name + '/adjoint.pth'))
    hnet.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_dynamics.pth'))
    
## Helper loss fct ##
# kl_loss btw Gaussian posterior distribution Q(z|X)=N(mu, logvar) and prior Gaussian distribution P(z)
def kl_loss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

Data = namedtuple('Data', ('q', 'p', 'u', 'f', 'r'))
#Terminal = namedtuple('Terminal', ('pt', 'nabla_qt'))

# Replay memory object
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        if len(self.memory) == self.capacity:
            random.shuffle(self.memory)
        self.memory.append(Data(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Use NeuralODE on hnet_target to sample N trajectories and update replay memory.
# Tuples in replay memory consists of (q, p, u, f(q, u), r(q, u))
def sample_step(q, p, env, HDnet, times, memory, control_coef, stochastic, device):
    qp = torch.cat((q, p), axis=1).to(device)
    with torch.no_grad():
        if stochastic:
            qps = sdeint(HDnet, qp, times)
        else:
            qps = odeint(HDnet, qp, times)
    # Go over each time-datapoint in the trajectory to update replay memory
    for i in range(qps.shape[0]):
        qpi_np = qps[i].cpu().detach().numpy()
        qi_np, pi_np = np.split(qpi_np, 2, axis=1)
        # Clipping if things are stochastic
        if stochastic:
            qi_np, pi_np = np.clip(qi_np, -MAX_VAL, MAX_VAL), np.clip(pi_np, -MAX_VAL, MAX_VAL)
        # Calculate u based on PMP condition H_u = 0
        u = (1.0/(2*control_coef))*np.einsum('ijk,ij->ik', env.f_u(qi_np), -pi_np)
        # Store info into a tuple for replay memory
        dynamic = env.f(qi_np, u); reward = env.L(qi_np, u)
        for j in range(qi_np.shape[0]):
            memory.push(torch.tensor(qi_np[j:(j+1), :], dtype=torch.float, device=device), 
                torch.tensor(pi_np[j:(j+1), :], dtype=torch.float, device=device), 
                torch.tensor(u[j:(j+1), :], dtype=torch.float, device=device), 
                torch.tensor(dynamic[j:(j+1), :], dtype=torch.float, device=device), 
                torch.tensor(reward[j:(j+1)], dtype=torch.float, device=device))

# Take (batch of) samples from replay memory and update reduced hamiltonian net (Use Huber instead of L2 loss)
def fit_hnet(memory, hnet, optim_hnet, batch_size):
    if len(memory) < batch_size:
            return 0
    data = memory.sample(batch_size)
    batch = Data(*zip(*data))
    q = torch.cat(batch.q)
    p = torch.cat(batch.p)
    f = torch.cat(batch.f)
    r = torch.cat(batch.r)
    qp = torch.cat((q, p), axis=1)

    # Compute Huber loss between reduced Hamiltonian and expected reduced Hamiltonian(instead of L1-loss)
    h_predict = hnet(qp).reshape(-1)
    h_expected =  (torch.einsum('ik,ik->i', p, f) + r)
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(h_predict, h_expected)
    # Optimize model
    optim_hnet.zero_grad()
    loss.backward()
    for param in hnet.parameters():
        param.grad.data.clamp_(-1, 1)
    optim_hnet.step()

    return loss

def train_hnet(stochastic, sigma, device, env, num_episodes, memory, adj_net, hnet, hnet_target, 
                T_end=5.0, n_timesteps=50, control_coef=0.5, use_adj_net=False, 
                update_interval=10, rate=1, batch_size_sample=256, batch_size=32, 
                num_hnet_train_max=1000000, stop_train_condition=0.001, lr=1e-3, log_interval=1000):
        # Load to device (GPU)
    if use_adj_net: 
        adj_net = adj_net.to(device); 
    hnet = hnet.to(device); hnet_target = hnet_target.to(device)
    hnet.reset_noise()
    # HDnet calculate the Hamiltonian dynamics network given the Hamiltonian target network
    if stochastic:
        HDnet = HDStochasticNet(Hnet=hnet_target, sigma=sigma, device=device).to(device)
    else:
        HDnet = HDNet(Hnet=hnet_target).to(device)
    # Optimizers for Hnet and AdjointNet
    #optim_hnet = torch.optim.Adam(hnet.parameters(), lr=lr)
    optim_hnet = torch.optim.SGD(hnet.parameters(), lr=lr, momentum=.9, nesterov=False)
    optim_hnet.zero_grad()
    # Times at which we sample data-points for each trajectory
    times = list(np.linspace(0, T_end + 1e-5, n_timesteps))
    times = torch.tensor(times, device=device, requires_grad=False)
    # qs are starting states of trajectory
    print('\nSampling while optimizing Hamiltonian net...')
    num_batch_samples = num_episodes//batch_size_sample
    iter = 0; total_loss = 0; cnt = 0
    while cnt < num_batch_samples:
        if iter%update_interval == 0 and cnt < num_batch_samples:
            # Copy parameters from hnet to hnet_target
            HDnet.copy_params(hnet)
            # Sample trajectories
            q = torch.tensor(env.sample_q(batch_size_sample), dtype=torch.float) #qs[cnt*batch_size_sample:(cnt+1)*batch_size_sample,:]
            if use_adj_net:
                p = adj_net(q.to(device)).cpu()
            else:
                p = torch.rand(q.shape, dtype=torch.float)-0.5
            sample_step(q, p, env, HDnet, times, memory, control_coef, stochastic, device)
            cnt += 1
            update_interval = int(update_interval*rate)
        # Train hnet at the same time to get better sampling
        loss_h = fit_hnet(memory, hnet, optim_hnet, batch_size)
        total_loss += loss_h
        if iter % log_interval == log_interval-1:
            print('\nIter {}: Average loss for (pretrained) reduced Hamiltonian network: {:.3f}'.format(iter+1, total_loss/log_interval))
            wandb.log({"HNet Loss": total_loss/log_interval})
            total_loss = 0
        iter += 1
    # Additional training for reduced Hamiltonian
    print('\nDone sampling. Now perform additional training for Hamiltonian net...')
    iter = 0; total_loss = 0
    while iter < num_hnet_train_max:
        loss_h = fit_hnet(memory, hnet, optim_hnet, batch_size)
        total_loss += loss_h
        if iter % log_interval == log_interval-1:
            print('\nIter {}: Average loss for reduced Hamiltonian network: {:.3f}'.format(iter+1, total_loss/log_interval))
            wandb.log({"HNet Loss": total_loss/log_interval})
            if iter > LEAST_NUM_TRAIN*log_interval and (total_loss/log_interval) < stop_train_condition:
                break
            total_loss = 0
        iter += 1
    print('\nDone training for Hamiltonian net.')

def sample_generator(qs, batch_size, shuffle=True):
    index = 0
    # initialize the list that will contain the current batch
    cur_batch = []
    # Number of data in qs
    num_q = qs.shape[0]
    # create index array
    data_index = [*range(num_q)]
    # shuffle line indexes if shuffle is set to True
    if shuffle:
        random.shuffle(data_index)
    # Infinite loop for generating samples
    while True:      
        if index >= num_q:
            # Reset the index
            index = 0
            if shuffle:
                random.shuffle(data_index)  
        q = qs[data_index[index]]
        cur_batch.append(q.reshape(1, -1))
        index += 1

        # if enough sample, then export them and reset cur_batch tmp storage
        if len(cur_batch) == batch_size:
            yield torch.cat(cur_batch, axis=0)       
            cur_batch = []

# Fit AdjointNet (minimize |pT|)
def fit_adjoint(q, times, adj_net, HDnet, optim_adj, stochastic, device):
    zero_tensor = torch.zeros(q.shape[0], q.shape[1], dtype=torch.float, device=device)
    criterion = torch.nn.SmoothL1Loss()
    p = adj_net(q.to(device))
    qp = torch.cat((q.to(device), p), axis=1)
    if stochastic:
        qps = sdeint(HDnet, qp, times)
    else:
        qps = odeint_adjoint(HDnet, qp, times, method='rk4', options={'step_size': 0.1})
 
    _, pt = torch.chunk(qps[-1], 2, axis=1)
    if stochastic:
        pt = torch.clip(pt, -MAX_VAL, MAX_VAL) # Clipping if things are stochastic
    #nabla_qi = torch.tensor(env.nabla_g(qi.cpu().detach().numpy()), dtype=torch.float, device=device)
    loss = criterion(pt, zero_tensor)
    
    optim_adj.zero_grad()
    loss.backward()
    for param in adj_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optim_adj.step()

    return loss

def train_adjoint(stochastic, sigma, device, env, num_episodes, adj_net, hnet,
                T_end=2.0, batch_size=64, lr=1e-3, log_interval=500,
                num_adj_train_max=1000, stop_train_condition=0.001):     
    # Setup HDnet, adjoint_net and optimizers
    if stochastic:
        HDnet = HDStochasticNet(Hnet=hnet, sigma=sigma, device=device).to(device)
    else:
        HDnet = HDNet(Hnet=hnet).to(device)
    adj_net = adj_net.to(device)
    adj_net.reset_noise()
    optim_adj = torch.optim.Adam(adj_net.parameters(), lr=lr)
    optim_adj.zero_grad()
    # Sample data qs for training adjoint net and times
    qs = torch.tensor(env.sample_q(num_episodes), dtype=torch.float)
    generator = sample_generator(qs, batch_size)
    times = list(np.linspace(0, T_end, 2))
    times = torch.tensor(times, device=device, requires_grad=True)
    # Now train the adjoint net
    print('\nAdjoint net training...')
    HDnet.copy_params(hnet) # Copy parameters from hnet to hnet_target
    total_loss = 0
    iter = 0; total_loss = 0
    while iter < num_adj_train_max:
        loss_adj = fit_adjoint(next(generator), times, adj_net, HDnet, optim_adj, stochastic, device)
        total_loss += loss_adj
        if iter % log_interval == log_interval-1:
            print('\nIter {}: Average loss for the adjoint network: {:.3f}'.format(iter+1, total_loss/log_interval))
            wandb.log({"Adjoint Net Loss": total_loss/log_interval})
            if iter > LEAST_NUM_TRAIN*log_interval and (total_loss/log_interval) < stop_train_condition:
                break
            total_loss = 0
        iter += 1

    print('\nDone adjoint net training.')
    
def training(stochastic, sigma, device, env, env_name, 
    adj_net, hnet, hnet_target, 
    num_train=10, num_warmup=10, load_model=False,
    T_hnet=5.0, T_adj=2.0, n_timesteps=50, control_coef=0.5,
    num_episodes_hnet=1024, num_episodes_adj=2048, 
    update_interval=10, rate=1, 
    batch_size_hnet_sample=256, batch_size_hnet=32, batch_size_adj=64,  
    lr_hnet=1e-3, lr_adj=1e-3, log_interval_hnet=1000, log_interval_adj=100,
    num_hnet_train_max=1000000, num_adj_train_max=1000, stop_train_condition=0.001,
    mem_capacity=1000000):
    r"""
    PMP training procedure with different types of modes. Currently only focus on first phase training
    Args:
        stochastic, sigma (bool): Whether to use stochastic dynamical system
        device (str): GPU or CPU
        env (str): Environment
        env_name (str): Used for saving and loading model 
        adj_net, hnet, hnet_target: networks to be traineed
        warmup: whether to do warm up phase or not
        load_model: Whether to load model to continue to train on this one
        num_train: Number of training phases after warmup phase
        T_hnet, T_adj: terminal times for adj_net and hnet
        n_timesteps: Number of time steps in the trajectory
        control_coef: coefficient c of control term cu^2 in the Lagrangian l(q, u) = cu^2 + l1(q)
        num_episodes_hnet, num_episodes_adj: Number of episodes to be sampled for training hnet and adj_net in each phase
        update_interval: Number of times hnet is minimized per sample step
        rate: the rate at which update_interval is increased
        batch_size_hnet_sample, batch_size_hnet, batch_size_adj: batch sizes for sampling and training
        lr_hnet, lr_adj: learning rates for Hnet and AdjointNet trainings
        log_interval_hnet, log_interval_adj: Record training losses interval
        num_hnet_train_max, num_adj_train_max: 
        stop_train_condition: loss fct condition to stop training
    """

    start_time = time.time()
    print('\nBegin training...')
    # load the models from files if needed
    if load_model:
        hnet.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_dynamics.pth'))
        adj_net.load_state_dict(torch.load('models/' + env_name + '/adjoint.pth'))
        print('Loaded Hamiltonian net and adjoint net from files')

    # Initalize replay memory
    memory = ReplayMemory(capacity=mem_capacity)

    print('\nWarm up training...')
    # Warm up step: first train hnet and then adj net
    for i in range(num_warmup):
        print(f'\nWarm up phase {i}:')
        train_hnet(stochastic, sigma, device, env, num_episodes_hnet, memory, adj_net, hnet, hnet_target, 
            T_end=T_hnet, n_timesteps=n_timesteps, control_coef=control_coef, use_adj_net=False, 
            update_interval=update_interval, rate=rate, batch_size_sample=batch_size_hnet_sample, 
            batch_size=batch_size_hnet, num_hnet_train_max=num_hnet_train_max, stop_train_condition=stop_train_condition, 
            lr=lr_hnet, log_interval=log_interval_hnet)
        train_adjoint(stochastic, sigma, device, env, num_episodes_adj, adj_net, hnet,
            T_end=T_adj, batch_size=batch_size_adj, lr=lr_adj, log_interval=log_interval_adj,
            num_adj_train_max=num_adj_train_max, stop_train_condition=stop_train_condition)

    print(f'\nDone warmup training. Main training for {num_train} steps')
    for i in range(num_train):
        print(f'\nMain phase {i}:')
        train_hnet(stochastic, sigma, device, env, num_episodes_hnet, memory, adj_net, hnet, hnet_target, 
            T_end=T_hnet, n_timesteps=n_timesteps, control_coef=control_coef, use_adj_net=True, 
            update_interval=update_interval, rate=1, batch_size_sample=batch_size_hnet_sample, 
            batch_size=batch_size_hnet, num_hnet_train_max=num_hnet_train_max, stop_train_condition=stop_train_condition, 
            lr=lr_hnet, log_interval=log_interval_hnet)
        train_adjoint(stochastic, sigma, device, env, num_episodes_adj, adj_net, hnet,
            T_end=T_adj, batch_size=batch_size_adj, lr=lr_adj, log_interval=log_interval_adj,
            num_adj_train_max=num_adj_train_max, stop_train_condition=stop_train_condition)

    save_models(adj_net, hnet, env_name)

    print('\nDone training. Training time is {:.4f} minutes'.format((time.time()-start_time)/60))