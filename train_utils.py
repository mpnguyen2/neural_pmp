from collections import namedtuple, deque
import random

import numpy as np
import pandas as pd

import torch
from torchdiffeq import odeint_adjoint as odeint

from common.common_nets import Mlp, Encoder
from model_nets import HDNet, HDVAE, HDInverseNet
from envs.classical_controls import MountainCar, CartPole, Pendulum
from envs.density_optimization import DensityOpt, DensityOptBoundary

def toList(s):
    tokens = s[1:-1].split(", ")
    ans = []
    for token in tokens:
        ans.append(int(token))
    return ans

## Reading architecture and training parameters routines ##
# Get correct environment
def get_environment(env_name):
    if env_name == 'mountain_car':
        return MountainCar()
    if env_name == 'cartpole':
        return CartPole()
    if env_name == 'pendulum':
        return Pendulum()
    if env_name == 'shape_opt':
        return DensityOpt()
    if env_name == 'shape_opt_boundary':
        return DensityOptBoundary()
    
# Get architecture
def get_architectures(arch_file, env_name, phase2=False):
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
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=adj_net_layer_dims, activation='relu')
    hnet = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=hnet_layer_dims)
    hnet_target = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=hnet_layer_dims)
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
    T, n_timesteps = info['T'].values[0], info['n_timesteps'].values[0]
    # Get control coefficients 
    control_coef = info['control_coef'].values[0]
    # Get training details for first phase (batch_size, num_epoch, lr, log_interval)
    # Currently refocus on first phase
    batch_size, lr_hnet, lr_adj, update_interval, log_interval = info['batch_size'].values[0], info['lr_hnet'].values[0],\
        info['lr_adj'].values[0], info['update_interval'].values[0], info['log_interval'].values[0]

    return float(T), int(n_timesteps), control_coef, int(batch_size), lr_hnet, lr_adj, int(update_interval), int(log_interval)

def get_test_params(param_file, env_name):
    pass

## Saving models ##
# save model phase 1
def save_models_phase1(AdjointNet, Hnet, env_name):
    torch.save(AdjointNet.state_dict(), 'models/' + env_name + '/adjoint.pth')
    torch.save(Hnet.state_dict(), 'models/' + env_name + '/hamiltonian_dynamics.pth')

# load model phase 1
def load_models_phase1(AdjointNet, Hnet, env_name):
    AdjointNet.load_state_dict(torch.load('models/' + env_name + '/adjoint.pth'))
    Hnet.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_dynamics.pth'))
    
# save model phase 2
def save_models_phase2(HnetDecoder, z_encoder, z_decoder, env_name):
    torch.save(HnetDecoder.state_dict(), 'models/' + env_name + '/hamiltonian_decoder.pth')
    torch.save(z_encoder.state_dict(), 'models/' + env_name + '/z_encoder.pth')
    torch.save(z_decoder.state_dict(), 'models/' + env_name + '/z_decoder.pth')
    
# load model phase 1
def load_models_phase2(HnetDecoder, z_encoder, z_decoder, env_name):
    HnetDecoder.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_decoder.pth'))
    z_encoder.load_state_dict(torch.load('models/' + env_name + '/z_encoder.pth'))
    z_decoder.load_state_dict(torch.load('models/' + env_name + '/z_decoder.pth'))
    
## Helper loss fct ##
# kl_loss btw Gaussian posterior distribution Q(z|X)=N(mu, logvar) and prior Gaussian distribution P(z)
def kl_loss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

Transition = namedtuple('Transition', ('q', 'p', 'u', 'f', 'r'))

# Replay memory object
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Use NeuralODE on Hnet_target to sample N trajectories and update replay memory.
# Tuples in replay memory consists of (q, p, u, f(q, u), r(q, u))
def sample_step(q, env, HDnet, times, memory, control_coef, device):
    q_np = q.detach().numpy()
    p = torch.rand(q.shape, dtype=torch.float)-0.5
    p_np = p.detach().numpy()
    qp = torch.cat((q, p), axis=1).to(device)
    qps = odeint(HDnet, qp, torch.tensor(times, device=device, requires_grad=True))
    # Go over each time-datapoint in the trajectory to update replay memory
    for i in range(qps.shape[0]):
        qpi_np = qps[i].cpu().detach().numpy()
        qi_np, pi_np = np.split(qpi_np, 2, axis=1)
        u = (1.0/(2*control_coef))*np.einsum('ijk,ij->ik', env.f_u(qi_np), -pi_np)
        dynamic = env.f(q, u); reward = env.L(q, u)
        memory.push(torch.tensor(qi_np, dtype=torch.float, device=device), torch.tensor(pi_np, dtype=torch.float, device=device), 
            torch.tensor(u, dtype=torch.float, device=device), torch.tensor(dynamic, dtype=torch.float, device=device), 
            torch.tensor(reward, dtype=torch.float, device=device))

# Take (batch of) samples from replay memory and update reduced hamiltonian net (Use Huber instead of L2 loss)
def fit_Hnet(memory, Hnet, optim_hnet, batch_size):
    if len(memory) < batch_size:
            return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    q = torch.cat(batch.q)
    p = torch.cat(batch.p)
    f = torch.cat(batch.f)
    r = torch.cat(batch.r)
    qp = torch.cat((q, p), axis=1)
    # Compute Huber loss between reduced Hamiltonian and expected reduced Hamiltonian(instead of L1-loss)
    h_predict = Hnet(qp).reshape(-1)
    h_expected =  (torch.einsum('ik,ik->i', p, f) + r)
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(h_predict, h_expected)
    # Optimize model
    loss.backward()
    optim_hnet.zero_grad()
    #for param in Hnet.parameters():
    #    param.grad.data.clamp_(-1, 1)
    optim_hnet.step(); 

    return loss

# Fit AdjointNet (minimize |pT|)
def fit_adjoint(device, AdjointNet, HDnet, memory, times, optim_adj, batch_size):
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    q = torch.cat(batch.q)
    p = AdjointNet(q)
    qp = torch.cat((q, p), axis=1)
    qps = odeint(HDnet, qp, torch.tensor(times, device=device, requires_grad=True))
    _, pt = torch.chunk(qps[-1], 2, axis=0)
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(pt, torch.zeros(pt.shape, device=device, dtype=torch.float))
    optim_adj.zero_grad()
    loss.backward()
    #for param in AdjointNet.parameters():
    #    param.grad.data.clamp_(-1, 1)
    optim_adj.step()

    return loss

## Main training procedure ##
def train_phase_1(device, env, num_trajs, AdjointNet, Hnet, Hnet_target, 
                T_end = 5.0, n_timesteps=50, control_coef=0.5,
                batch_size=32, update_interval=10, rate=2, mem_capacity=10000,
                num_hnet_train_max=40000, num_adjoint_train_max=1000, stop_train_condition=0.001,
                lr_hnet=1e-3, lr_adj=1e-3, log_interval=50):
    # Load to device (GPU)
    AdjointNet = AdjointNet.to(device); Hnet = Hnet.to(device); Hnet_target = Hnet_target.to(device)
    # HDnet calculate the Hamiltonian dynamics network given the Hamiltonian target network
    HDnet = HDNet(Hnet=Hnet_target).to(device)
    # Optimizers for Hnet and AdjointNet
    optim_hnet = torch.optim.Adam(Hnet.parameters(), lr=lr_hnet)
    optim_adj = torch.optim.Adam(AdjointNet.parameters(), lr=lr_adj)
    optim_hnet.zero_grad(); optim_adj.zero_grad()
    # Times at which we sample data-points for each trajectory
    times = list(np.linspace(0, T_end + 1e-5, n_timesteps))
    # replay memory
    memory = ReplayMemory(capacity=mem_capacity)
    # qs are starting states of trajectory and cnt is the number of qs used
    qs = torch.tensor(env.sample_q(num_trajs), dtype=torch.float); cnt = 0
    iter = 0; total_loss = 0
    while cnt < num_trajs:
        if iter%update_interval == 0 and cnt < num_trajs:
            # Copy parameters from Hnet to Hnet_target
            HDnet.copy_params(Hnet)
            # Sample trajectories
            sample_step(qs[cnt:(cnt+1),:], env, HDnet, times, memory, control_coef, device)
            cnt += 1
            update_interval = int(update_interval*rate)
        # Train Hnet at the same time to get better sampling
        loss_h = fit_Hnet(memory, Hnet, optim_hnet, batch_size)
        total_loss += loss_h
        if iter % log_interval == 0:
            print('\nIter {}: Average loss for (pretrained) reduced Hamiltonian network: {:.3f}'.format(iter+1, total_loss/(log_interval*batch_size)))
            total_loss = 0
        iter += 1
    # Additional training for reduced Hamiltonian
    iter = 0; total_loss = 0
    while iter < num_hnet_train_max and (iter < num_hnet_train_max/2 or total_loss < stop_train_condition):
        loss_h = fit_Hnet(memory, Hnet, optim_hnet, batch_size)
        total_loss += loss_h
        if iter % log_interval == 0:
            print('\nIter {}: Average loss for reduced Hamiltonian network: {:.3f}'.format(iter+1, total_loss/(log_interval*batch_size)))
            total_loss = 0
        iter += 1
    # Finally we train the adjoint net
    iter = 0; total_loss = 0
    while iter < num_adjoint_train_max and (iter < num_adjoint_train_max/2 or total_loss < stop_train_condition):
        loss_adj = fit_adjoint(device, AdjointNet, HDnet, memory, times, optim_adj, batch_size)
        total_loss += loss_adj
        if iter % log_interval == 0:
            print('\nIter {}: Average loss for adjoint network: {:.3f}'.format(iter+1, total_loss/(log_interval*batch_size)))
            total_loss = 0
        iter += 1

# Temporary wait not train phase 2 yet  
def train_phase_2(AdjointNet, Hnet, HnetDecoder, z_decoder, z_encoder, qs, 
                  T2 = 1.0, beta = 1.0, 
                  batch_size=32, num_epoch=20, lr=1e-3, 
                  log_interval=50, env_name=''):
    
    HDVAE_net = HDVAE(AdjointNet, Hnet, HnetDecoder, z_encoder, z_decoder, T2)
    # Optimizer for Hamiltonian net decoder, (additional) latent encoder and decoder
    optim = torch.optim.Adam(list(HnetDecoder.parameters()) + 
                             list(z_encoder.parameters()) +
                             list(z_decoder.parameters()), lr=lr)
    optim.zero_grad()
    
    # Training over the same data qs num_epoch epochs
    num_samples = qs.shape[0]
    for i in range(num_epoch):
        print('\nEpoch {}: '.format(i+1))
        loss = 0
        q_dat = torch.clone(qs)[torch.randperm(num_samples)]
        num_iter = q_dat.shape[0]//batch_size
        total_loss = 0
        for j in range(num_iter):
            # state training examples
            q = q_dat[j*batch_size:(j+1)*batch_size]
            # Hamiltonian VAE net returns starting coupled state (state+adjoint)
            # terminal coupled state and its construction, starting state construction
            # mean and logvar of the actual latent variable mapped from terminal state 
            qp, qp_hat, qpt, qpt_hat, mu, logvar = HDVAE_net(q)
            # Reconstruction loss
            loss = torch.sum((qp-qp_hat)**2) + torch.sum((qpt-qpt_hat)**2) # + KL based on mu logvar
            # KL loss
            loss += beta * kl_loss(mu, logvar)
            # Optim step
            loss.backward()
            optim.step(); optim.zero_grad()
            # Print progress
            total_loss += loss.item()
            if j % log_interval == 0:
                print('Average loss for {}th iteration is: {}'.format(j+1, total_loss/((j+1)*batch_size)))
    
def get_extreme_samples(env, AdjointNet, Hnet, HnetDecoder, z_decoder, z_encoder, 
                        qs, T=1, num_samples_per_seed=50):
    HDnet = HDNet(Hnet=Hnet)
    q_dim = env.q_dim
    #seed_z = []
    q_samples = []
    with torch.no_grad():
        for q in qs:
            q = q.reshape(1, -1)
            p = AdjointNet(q)
            qp = torch.cat((q, p), axis=1)
            times = []
            s = 0.0; max_num_step=200
            for i in range(max_num_step):
                times.append(s)
                s += 0.5
            qp_traj = odeint(HDnet, qp, torch.tensor(times, requires_grad=True))
            for i in range(max_num_step-1):
                if env.criteria_q(qp_traj[i].detach().numpy()[0, :q_dim])\
                       < env.criteria_q(qp_traj[i+1].detach().numpy()[0, :q_dim]):
                           seed_qp = qp_traj[i]
                           q_samples.append(seed_qp.detach().numpy()[0, :q_dim])
                           #seed_z.append(z_encoder(seed_qp))
                           break
        
        return np.array(q_samples)
    
def training(device, env, env_name, num_trajs,
    AdjointNet, Hnet, Hnet_target,
    T=5.0, n_timesteps=50, control_coef=0.5,
    batch_size=32, update_interval=10, rate=2,
    num_hnet_train_max=40000, num_adjoint_train_max=1000, stop_train_condition=0.001,
    lr_hnet=1e-3, lr_adj=1e-3, log_interval=1,
    retrain_phase1=True):
    """
    PMP training procedure with different types of modes. Currently only focus on first phase training
    Args:
        AdjointNet, Hnet: networks to be traineed
        T: terminal times of phase 1
        control_coef: coefficient c of control term cu^2 in the Lagrangian l(q, u) = cu^2 + ...
        batch_size: batch size for Hamiltonian net training
        num_trajs: Number of trajectories to be sampled
        lr_hnet, lr_adj: learning rates for Hnet and AdjointNet trainings
        update_interval: When to update target hamiltonian net
        log_interval: Record training losses interval
    """
    
    if retrain_phase1:
        print('\nTraining phase 1...')
        train_phase_1(device, env, num_trajs, AdjointNet, Hnet, Hnet_target, 
                  T_end = T, n_timesteps=n_timesteps, control_coef=control_coef,
                  batch_size=batch_size, update_interval=update_interval, rate=rate,
                  num_hnet_train_max=num_hnet_train_max, num_adjoint_train_max=num_adjoint_train_max, 
                  stop_train_condition=stop_train_condition,
                  lr_hnet=lr_hnet, lr_adj=lr_adj, log_interval=log_interval)
            
    else:
        load_models_phase1(AdjointNet, Hnet, env_name)
        print('\nLoaded phase 1 trained models (adjoint net and Hamiltonian net).\n')

    # Save models
    print('\nDone training. Saving model...')
    save_models_phase1(AdjointNet, Hnet, env_name)
    print('Model saved!')