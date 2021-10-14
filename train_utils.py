import numpy as np

import torch
from torchdiffeq import odeint_adjoint as odeint

from common.common_nets import Mlp
from model_nets import HDNet, HDVAE
from envs.classical_controls import ContinuousEnv

# kl_loss btw Gaussian posterior distribution Q(z|X)=N(mu, logvar) and prior Gaussian distribution P(z)
def kl_loss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

def train_phase_1(env, AdjointNet, Hnet, qs, 
                  T1 = 1.0, scale=0.5, dynamic_hidden=False, alpha = 1, 
                  batch_size=32, num_epoch=20, lr=1e-3, 
                  log_interval=50, model_name=''):
    # HDnet calculate the Hamiltonian dynamics network given the Hamiltonian network Hnet
    HDnet = HDNet(Hnet=Hnet)
    # Optimizer for HDnet
    optim = torch.optim.Adam(list(Hnet.parameters()) + list(AdjointNet.parameters()), lr=lr)
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
        num_iter = q_dat.shape[0]//batch_size
        total_loss = 0
        for j in range(num_iter):
            # state training examples
            q = q_dat[j*batch_size:(j+1)*batch_size]
            q_np = q.detach().numpy()
            #print('q', q.shape)
            # adjoint and generalized coordinates qp = (q, p)
            p = AdjointNet(q)
            #print('p', p.shape)
            p_np = p.detach().numpy()
            qp = torch.cat((q, p), axis=1)
            #print('qp', qp.shape)
            # Given the starting generalized coordinate, use reduced hamiltonian to get 
            # the ending generalized coordinate
            qp_t = odeint(HDnet, qp, torch.tensor(times, requires_grad=True))[0]
            qt, pt = torch.chunk(qp_t, 2, dim=1)
            qt_np = qt.detach().numpy()
            #print('qt', qt.shape)
            ## Loss function = (pt - nabla g(qt))**2 + alpha * (h(q, p) - ((p, f(q, u)) + L(q, u))
            ## First part require nabla g(qt): (pt - nabla g(qt))**2
            dg = torch.tensor(env.nabla_g(qt_np))
            #print('nabla g', dg.shape)
            loss = torch.sum((pt-dg)**2) 
            ## Second part of loss function
            # Calculate optimal u = -p^T f_u(q, u) (based on adjoint)
            u = (1.0/(2*scale))*np.einsum('ijk,ij->ik', env.f_u(q_np), -p_np)
            #print('u', u.shape)
            if dynamic_hidden:
                ## (p, f(q, u)) + L(q, u) = (p, qdot_np) + L(q, u)
                # Calculate qdot_np
                qp_dot = HDnet(0, qp)
                qdot, pdot = torch.chunk(qp_dot, 2, dim=1)
                #print('qdot, pdot', qdot.shape, pdot.shape)
                qdot_np = qdot.detach().numpy()
                # Calculate reference reduced Hamiltonian using usual Hamiltonian but with supposedly optimal control
                h_pq_ref = np.einsum('ik,ik->i', p_np, qdot_np) + env.L(q_np, u)
            else:
                h_pq_ref = np.einsum('ik,ik->i', p_np, env.f(q_np, u)) + env.L(q_np, u)
            #print('h_pq_ref', h_pq_ref.shape)
            h_pq = Hnet(qp)
            #print('h_pq', h_pq.shape)
            loss += alpha*torch.sum((h_pq-torch.tensor(h_pq_ref))**2)
            # optimize step
            loss.backward()
            optim.step(); optim.zero_grad()
            # Print progress
            total_loss += loss.item()    
            if j % log_interval == 0:
                print('Average loss for {}th iteration is: {}'.format(j+1, total_loss/((j+1)*batch_size)))
    
    # save model
    torch.save(AdjointNet.state_dict(), 'models/' + model_name + '/adjoint.pth')
    torch.save(Hnet.state_dict(), 'models/' + model_name + '/hamiltonian_dynamics.pth')
    
def train_phase_2(AdjointNet, Hnet, HnetDecoder, z_decoder, z_encoder, qs, 
                  T2 = 1.0, beta = 1.0, 
                  batch_size=32, num_epoch=20, lr=1e-3, 
                  log_interval=50, model_name=''):
    AdjointNet.load_state_dict(torch.load('models/' + model_name + '/adjoint.pth'))
    Hnet.load_state_dict(torch.load('models/' + model_name + '/hamiltonian_dynamics.pth'))
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

    # save model
    torch.save(HnetDecoder.state_dict(), 'models/' + model_name + '/hamiltonian_decoder.pth')
    torch.save(z_encoder.state_dict(), 'models/' + model_name + '/z_encoder.pth')
    torch.save(z_decoder.state_dict(), 'models/' + model_name + '/z_decoder.pth')