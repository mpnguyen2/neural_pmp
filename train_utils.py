import numpy as np
import pandas as pd

import torch
from torchdiffeq import odeint_adjoint as odeint

from common.common_nets import Mlp, Encoder
from model_nets import HDNet, HDVAE, HDInverseNet
from envs.classical_controls import MountainCar, CartPole, Pendulum

## Reading architecture and training parameters routines ##
def toList(s):
    tokens = s[1:-1].split(", ")
    ans = []
    for token in tokens:
        ans.append(int(token))
    return ans

# Get correct environment
def get_environment(env_name):
    if env_name == 'mountain_car':
        return MountainCar()
    if env_name == 'cartpole':
        return CartPole()
    if env_name == 'pendulum':
        return Pendulum()

# Get architecture
def get_architectures(arch_file, env_name):
    # Get architecture info from arch_file
    df = pd.read_csv(arch_file)
    info = df[df['env_name']==env_name]
    # Extract architecture info
    q_dim=info['q_dim'].values[0]
    adj_net_layer_dims = toList(info['adj_net_layer_dims'].values[0])
    hnet_layer_dims = toList(info['hnet_layer_dims'].values[0])
    hnet_decoder_layer_dims = toList(info['hnet_decoder_layer_dims'].values[0])
    z_dim = int(info['z_dim'].values[0])
    z_encoder_share_layer_dims = toList(info['z_encoder_share_layer_dims'].values[0])
    z_encoder_mean_layer_dims = toList(info['z_encoder_mean_layer_dims'].values[0])
    z_encoder_logvar_layer_dims = toList(info['z_encoder_logvar_layer_dims'].values[0])
    z_decoder_layer_dims = toList(info['z_decoder_layer_dims'].values[0])
    # Build adjoint variable and Hamiltonian nets
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=adj_net_layer_dims)
    hnet = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=hnet_layer_dims)
    # Build hnet_decoder
    hnet_decoder = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=hnet_decoder_layer_dims)
    # Build latent z_encoder
    z_encoder = Encoder(input_dim=2*q_dim, share_layer_dims=z_encoder_share_layer_dims, 
        mean_layer_dims=z_encoder_mean_layer_dims, 
        logvar_layer_dims=z_encoder_logvar_layer_dims, 
        output_dim=z_dim)
    # Build latent z_decoder
    z_decoder = Mlp(input_dim=z_dim, output_dim=2*q_dim, layer_dims=z_decoder_layer_dims)
    
    return q_dim, adj_net, hnet, hnet_decoder, z_encoder, z_decoder
    
def get_train_params(param_file, env_name):
    # Get parameter info from param file
    df = pd.read_csv(param_file)
    info = df[df['env_name']==env_name]
    # Get terminal times T1, T2
    T1, T2 = info['T1'].values[0], info['T2'].values[0]
    # Get dynamic_hiden options and control coefficients 
    control_coef, dynamic_hidden = info['control_coef'].values[0], info['dynamic_hidden'].values[0]
    # Get hyperparameter loss function 
    alpha1, alpha2, beta1, beta2 = info['alpha1'].values[0], info['alpha2'].values[0], info['beta1'].values[0], info['beta2'].values[0]
    # Get training details for first phase (batch_size, num_epoch, lr, log_interval)
    batch_size1, num_epoch1, lr1, log_interval1 = info['batch_size1'].values[0],\
        info['num_epoch1'].values[0], info['lr1'].values[0], info['log_interval1'].values[0]
    # Get training details for second phase (batch_size, num_epoch, lr, log_interval)
    batch_size2, num_epoch2, lr2, log_interval2 = info['batch_size2'].values[0],\
        info['num_epoch2'].values[0], info['lr2'].values[0], info['log_interval2'].values[0]

    return T1, T2, control_coef, dynamic_hidden, alpha1, alpha2, beta1, beta2,\
        batch_size1, num_epoch1, lr1, log_interval1, batch_size2, num_epoch2, lr2, log_interval2

def get_test_params(param_file, env_name):
    pass

## Helper loss fct ##
# kl_loss btw Gaussian posterior distribution Q(z|X)=N(mu, logvar) and prior Gaussian distribution P(z)
def kl_loss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

## Main training procedure ##
def train_phase_1(env, AdjointNet, Hnet, qs, 
                  T1 = 1.0, control_coef=0.5, dynamic_hidden=False, 
                  alpha1=1, alpha2=0.1, beta=1, 
                  batch_size=32, num_epoch=20, lr=1e-3, 
                  log_interval=50):
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
            qp_t = odeint(HDnet, qp, torch.tensor(times, requires_grad=True))[-1]
            qt, pt = torch.chunk(qp_t, 2, dim=1)
            qt_np = qt.detach().numpy()
            #print('qt', qt.shape)
            ## Loss function = (pt - nabla g(qt))**2 + alpha * (h(q, p) - ((p, f(q, u)) + L(q, u))
            ## First part require nabla g(qt): (pt - nabla g(qt))**2
            dg = torch.tensor(env.nabla_g(qt_np))
            dg0 = torch.tensor(env.nabla_g(q_np))
            #print('nabla g', dg.shape)
            loss = alpha1*torch.sum((p-dg0)**2) + alpha2*torch.sum((pt-dg)**2)
            ## Second part of loss function
            # Calculate optimal u = -p^T f_u(q, u) (based on adjoint)
            u = (1.0/(2*control_coef))*np.einsum('ijk,ij->ik', env.f_u(q_np), -p_np)
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
            loss += beta*torch.sum((h_pq-torch.tensor(h_pq_ref))**2)
            # optimize step
            loss.backward()
            optim.step(); optim.zero_grad()
            # Print progress
            total_loss += loss.item()    
            if j % log_interval == 0:
                print('Average loss for {}th iteration is: {}'.format(j+1, total_loss/((j+1)*batch_size)))
    
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
    
    
def get_extreme_samples(env, AdjointNet, Hnet, HnetDecoder, z_decoder, z_encoder, 
                        qs, T=1, num_samples_per_seed=50):
    HDnet = HDNet(Hnet=Hnet)
    q_dim = env.q_dim
    seed_z = []
    with torch.no_grad():
        for q in qs:
            p = AdjointNet(q)
            qp = torch.cat((q, p), axis=1)
            times = [0]
            s = 0.05; max_num_step=20
            for i in range(max_num_step):
                times.append(s)
                s *= 2
            qp_traj = odeint(HDnet, qp, torch.tensor(times, requires_grad=True))
            for i in range(max_num_step):
                if env.criteria_q(qp_traj[i].detach().numpy()[:q_dim])\
                       > env.criteria_q(qp_traj[i+1].detach().numpy()[:q_dim]):
                           seed_qp = qp_traj[i]
                           seed_z.append(z_encoder(seed_qp))
                           break
    
        HDInversenet = HDInverseNet(HnetDecoder)
        q_samples = []
        times = [0.0, T]
        for mu, logvar in seed_z:
            std = torch.exp(0.5*logvar)
            for i in range(num_samples_per_seed):
                eps = torch.randn_like(std)
                qp_t = z_decoder(mu + eps*std)
                qp = odeint(HDInversenet, qp_t, torch.tensor(times, requires_grad=True))[-1]
                q_samples.append(qp.detach().numpy()[:q_dim])
        
        return np.array(q_samples)
    
def training(env, env_name, qs, 
    AdjointNet, Hnet, HnetDecoder, z_decoder, z_encoder, 
    T1=1, T2=1, control_coef=0.5, dynamic_hidden=False,
    alpha1=1, alpha2=0.1, beta1=1, beta2=1,
    batch_size1=32, batch_size2=32, num_epoch1=20, num_epoch2=20,
    lr1=1e-3, lr2=1e-3, log_interval1=50, log_interval2=50,
    mode=0, retrain_phase1=True, retrain_phase2=True,
    num_examples_phase2=0.2, num_additional_train=5):
    """
    PMP training procedure with different types of modes
    Args:
        AdjointNet, Hnet, HnetDecoder, z_decoder, z_encoder: networks to be traineed
        T1, T2: terminal times of phase 1 and 2
        control_coef: coefficient c of control term cu^2 in the Lagrangian l(q, u) = cu^2 + ...
        dynamic_hidden: whether to use dynamics f or use implicit qdot in certain calculations
        alpha1, alpha2, beta1, beta2: hyperparameters in current training loss function
        batch_size1, num_epoch1, lr1, log_interval1: training details for phase 1
        batch_size2, num_epoch2, lr2, log_interval2: training details for phase 2
        mode (int): specify whether we want to do a vanilla phase 1 training with mode=0, or 2-phase training with mode=1, 
                    or 2-phase training with extreme case sample efficient with mode=2
        num_examples_phase2 (float): the percentage amount of examples to be used in phase 2 (from phase 1)
        num_addition_train (int): number of additional training with both original data and new generated data from efficient scheme
        retrain_phase1, retrain_phase2: whether to retrain phase 1, phase 2
    """
    
    if retrain_phase1:
        print('\nTraining phase 1...')
        train_phase_1(env, AdjointNet, Hnet, qs, 
            T1, control_coef, dynamic_hidden, 
            alpha1, alpha2, beta=beta1, 
            batch_size=batch_size1, num_epoch=num_epoch1, lr=lr1, 
            log_interval=log_interval1)
    else:
        load_models_phase1(AdjointNet, Hnet, env_name)
        print('\nLoaded phase 1 trained models (adjoint net and Hamiltonian net).\n')
    
    if mode >= 1:
        if retrain_phase2:
            print('\nTraining phase 2...')
            num_examples = int(num_examples_phase2*qs.shape[0])
            qs2 = torch.clone(qs)[torch.randperm(qs.shape[0])][:num_examples]
            train_phase_2(AdjointNet, Hnet, HnetDecoder, z_decoder, z_encoder, qs2, 
                      T2, beta = beta1, batch_size=batch_size2, num_epoch=num_epoch2, 
                      lr=lr2, log_interval=log_interval2, env_name=env_name)
        else:
            load_models_phase2(HnetDecoder, z_encoder, z_decoder, env_name)
            print('\nLoaded phase 2 trained models (Hamiltonian decoder and latent encoder and decoder).\n')
        
    if mode == 2:
        qs_extreme_np = get_extreme_samples(env, AdjointNet, Hnet, HnetDecoder, z_decoder, z_encoder, 
                        qs, T=1, num_samples_per_seed=50)
        qs_extreme = torch.tensor(qs_extreme_np, dtype=torch.float)
        
        print('\nRetraining phase 1 with new data...')
        for i in range(num_additional_train):
            print(f'\nRetraining phase 1 with new data {i}th time...')
            # Training using extreme data
            train_phase_1(env, AdjointNet, Hnet, qs_extreme, 
                T1, control_coef, dynamic_hidden, 
                alpha1, alpha2, beta=beta1, 
                batch_size=batch_size1, num_epoch=num_epoch1, lr=lr1, 
                log_interval=log_interval1)
            # Training using usual data
            train_phase_1(env, AdjointNet, Hnet, qs, 
                T1, control_coef, dynamic_hidden, 
                alpha1, alpha2, beta=beta1, 
                batch_size=batch_size1, num_epoch=num_epoch1, lr=lr1, 
                log_interval=log_interval1)

    # Save models
    print('\nDone training. Saving model...')
    save_models_phase1(AdjointNet, Hnet, env_name)
    if mode >= 1:
        save_models_phase2(HnetDecoder, z_encoder, z_decoder, env_name)
    print('Done.')