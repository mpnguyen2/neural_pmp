import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint

from train_utils import train_phase_2
from common.common_nets import Mlp, Encoder
from model_nets import HDNet
from envs.classical_controls import Pendulum


# 5 10
a = torch.tensor([[1, 1, 2], [3, 4, 5]])
b = torch.tensor([[1, 0, 2], [2, 1, 0]])

c = torch.einsum('ij, ij->i', a, b)
#print(c.shape)
'''
env = Pendulum()

# models
q_dim=2; 
adj_net= Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
Hnet = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
# Build symplectic dynamics net from Hamiltonian net
HDnet = HDNet(Hnet=Hnet)

# Run optimal trajectory
#q = torch.tensor(env.sample_q(1, mode='test'), dtype=torch.float)
q = torch.zeros(1, 2)
p = adj_net(q)
qp = torch.cat((q, p), axis=1)
time_steps = list(np.arange(0, 10, 0.01))
traj = odeint(HDnet, qp, torch.tensor(time_steps, requires_grad=False))
print('Done finding trajectory...')
# Print trajectory and save images to vid
cnt = 0
for e in traj:
    qe, _ = torch.chunk(e, 2, dim=1)
    qe_np = qe.detach().numpy()
    env.render(qe_np.reshape(-1))
    cnt += 1

env.close()
'''

'''
q_dim=2; model_name='mountain_car'
adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
adj_net.load_state_dict(torch.load('models/' + model_name + '/adjoint.pth'))
h_net.load_state_dict(torch.load('models/' + model_name + '/hamiltonian_dynamics.pth'))
print('Adjoint net...')
for param in adj_net.parameters():
    print(param.data)

print('H net...')
for param in h_net.parameters():
    print(param.data)
'''

import pandas as pd

data = {'name': ['1a', '2b', '3c'],
        'price': [[0, 1, 2], [2, 3, 5], [3, 6, 8]],
        'isLocked':[True, False, False]}

df = pd.DataFrame(data, columns=['name', 'price', 'isLocked'])
df.to_csv('models/test_csv.csv', index=False, header=True)
df1 = pd.read_csv('models/test_csv.csv')
print(list(df1[df1['name']=='1a']['price'].values[0]))


'''
def train_backward_phase_1(env, Hnet, qs, 
                  T1 = 1.0,
                  alpha1=1,
                  batch_size=32, num_epoch=20, lr=1e-3, 
                  log_interval=50):
    # HDnet calculate the inverse Hamiltonian dynamics network given the Hamiltonian network Hnet
    HDnet = HDInverseNet(Hnet=Hnet)
    # Optimizer for HDnet
    optim = torch.optim.Adam(list(Hnet.parameters()), lr=lr)
    optim.zero_grad()
    # Go over each batch of data and optimize HDnet
    times = [0, T1]
    # Generate num_samples data qs consisting of only state (q) samples
    num_samples = qs.shape[0]
    # Training over the same data qs num_epoch epochs
    for i in range(num_epoch):
        print('\nEpoch {}: '.format(i+1))
        loss = 0
        q_dat = torch.clone(qs)[torch.randperm(num_samples)]
        total_loss = 0
        num_iter = num_samples//batch_size
        for j in range(num_iter):
            # state training examples
            #qt = q_dat #+ 0.01*torch.clamp(torch.randn(q_dat.shape), -1, 1)
            qt = q_dat[j*batch_size:(j+1)*batch_size]
            qt_np = qt.detach().numpy()
            #print('q', q.shape)
            #pt_np = nabla_qt_np
            pt = torch.tensor(env.nabla_g(qt_np), dtype=torch.float)
            #print('p', p.shape)
            qp_t = torch.cat((qt, pt), axis=1)
            #print('qp', qp.shape)
            # Given the starting generalized coordinate, use reduced hamiltonian to get 
            # the ending generalized coordinate
            qp = odeint(HDnet, qp_t, torch.tensor(times, requires_grad=True))[-1]
            q, p = torch.chunk(qp, 2, dim=1)          
            loss = torch.sum(p**2) 
            # optimize step
            loss.backward()
            optim.step(); optim.zero_grad()
            # Print progress
            total_loss += loss.item()
            if j % log_interval == 0:
                print('Average loss for {}th iteration is: {}'.format(j+1, total_loss/batch_size))
                total_loss = 0

def train_simple_phase_1(env, AdjointNet, qs, 
                  T1 = 1.0,
                  batch_size=32, num_epoch=20, lr=1e-3, 
                  log_interval=50):
    # HDnet calculate the inverse Hamiltonian dynamics network given the Hamiltonian network Hnet
    HDnet = HDSimpleNet()
    # Optimizer for HDnet
    #optim = torch.optim.SGD(list(AdjointNet.parameters()), lr=lr, momentum=0.9)
    optim = torch.optim.Adam(list(AdjointNet.parameters()), lr=lr)
    #optim = torch.optim.Adadelta(list(AdjointNet.parameters()), lr=lr)
    optim.zero_grad()
    # Go over each batch of data and optimize HDnet
    times = [0, T1]
    # Generate num_samples data qs consisting of only state (q) samples
    num_samples = qs.shape[0]
    # Training over the same data qs num_epoch epochs
    for i in range(num_epoch):
        print('\nEpoch {}: '.format(i+1))
        q_dat = torch.clone(qs)[torch.randperm(num_samples)]
        total_loss = 0
        num_iter = num_samples//batch_size
        for j in range(num_iter):
            # state training examples
            #qt = q_dat #+ 0.01*torch.clamp(torch.randn(q_dat.shape), -1, 1)
            qt = q_dat[j*batch_size:(j+1)*batch_size]
            qt_np = qt.detach().numpy()
            pt = torch.tensor(env.nabla_g(qt_np), dtype=torch.float)
         
            qp_t = torch.cat((qt, pt), axis=1)
            # Given the ending generalized coordinate, use backward reduced hamiltonian to get 
            # the starting generalized coordinate
            qp = odeint(HDnet, qp_t, torch.tensor(times, requires_grad=True))[-1]
            q, p = torch.chunk(qp, 2, dim=1)       
            q0 = torch.tensor(q.detach().numpy(), dtype=torch.float)
            p0 = torch.tensor(p.detach().numpy(), dtype=torch.float)
        
            loss = torch.mean((pt - AdjointNet(qt))**2) 
            # optimize step
            loss.backward()
            optim.step(); optim.zero_grad()
            # Print progress
            total_loss += loss.item()
        print('Average loss for {}th epoch is: {}'.format(i+1, total_loss/num_samples))
'''

'''
qpi_dot = HDnet(times[i], qp)
qi_dot, pi_dot = torch.chunk(qpi_dot, 2, dim=1)
#print('qdot, pdot', qdot.shape, pdot.shape)
qi_dot_np = qi_dot.detach().numpy()
# Calculate reference reduced Hamiltonian using usual Hamiltonian but with supposedly optimal control
h_pq_ref = np.einsum('ik,ik->i', pi_np, qi_dot_np) + env.L(qi_np, u)
'''

'''
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

'''