import numpy as np
import torch
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from model_nets import HDNet, HDVAE, HDInverseNet
import utils

# Get extreme samples for more focusing.
def get_extreme_samples(env, adj_net, hnet, hnet_decoder, z_decoder, z_encoder, 
                        qs, T=1, num_samples_per_seed=50):
    hd_net = HDNet(hnet=hnet)
    q_dim = env.q_dim
    seed_z = []
    with torch.no_grad():
        for q in qs:
            p = adj_net(q)
            qp = torch.cat((q, p), axis=1)
            times = [0]
            s = 0.05; max_num_step=20
            for i in range(max_num_step):
                times.append(s)
                s *= 2
            qp_traj = odeint(hd_net, qp, torch.tensor(times, requires_grad=True))
            for i in range(max_num_step):
                if env.criteria_q(qp_traj[i].detach().numpy()[:q_dim])\
                       > env.criteria_q(qp_traj[i+1].detach().numpy()[:q_dim]):
                           seed_qp = qp_traj[i]
                           seed_z.append(z_encoder(seed_qp))
                           break
        hd_inverse_net = HDInverseNet(hnet_decoder)
        q_samples = []
        times = [0.0, T]
        for mu, logvar in seed_z:
            std = torch.exp(0.5*logvar)
            for i in range(num_samples_per_seed):
                eps = torch.randn_like(std)
                qp_t = z_decoder(mu + eps*std)
                qp = odeint(hd_inverse_net, qp_t, torch.tensor(times, requires_grad=True))[-1]
                q_samples.append(qp.detach().numpy()[:q_dim])
        
        return np.array(q_samples)
    
# KL loss
def kl_loss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

# Train phase 1
def train_phase_1(env, adj_net, hnet, qs, 
                  T1=1.0, control_coef=0.5, dynamic_hidden=False, 
                  alpha1=1, alpha2=0.1, beta=1, 
                  num_epoch=20, num_iter=20, batch_size=32, lr=1e-3, 
                  log_interval=50):
    # HDnet calculate the Hamiltonian dynamics network given the Hamiltonian network Hnet
    hd_net = HDNet(hnet=hnet)
    # Optimizer for HDnet
    optim = torch.optim.Adam(list(hnet.parameters()) + list(adj_net.parameters()), lr=lr)
    optim.zero_grad()
    # Go over each batch of data and optimize HDnet
    times = [0, T1]
    # Generate num_samples data qs consisting of only state (q) samples
    num_samples = qs.shape[0]
    # Training over the same data qs num_epoch epochs
    for i in range(num_epoch):
        print('\nEpoch {}: '.format(i+1))
        loss = 0; cnt = 0
        q_dat = torch.clone(qs)[torch.randperm(num_samples)]
        total_loss = 0
        for j in range(num_iter):
            # state training examples
            q = q_dat[np.random.choice(num_samples, batch_size, replace=False)] #[j*batch_size:(j+1)*batch_size]
            q_np = q.detach().numpy()
            #print('q', q.shape)
            # adjoint and generalized coordinates qp = (q, p)
            p = adj_net(q)
            #print('p', p.shape)
            p_np = p.detach().numpy()
            qp = torch.cat((q, p), axis=1)
            #print('qp', qp.shape)

            # Given the starting generalized coordinate, use reduced hamiltonian to get 
            # the ending generalized coordinate
            qp_t = odeint(hd_net, qp, torch.tensor(times, requires_grad=True))[-1]
            qt, pt = torch.chunk(qp_t, 2, dim=1)
            qt_np = qt.detach().numpy()
            #print('qt', qt.shape)

            ## Loss function = alpha1(p0 - nabla g(q0))**2 + alpha2(pt - nabla g(qt))**2 + beta1 * (h(q, p) - ((p, f(q, u)) + L(q, u)))
            # First parts require nabla g(q0) and nabla g(qt)
            dg0 = torch.tensor(env.nabla_g(q_np))
            dg = torch.tensor(env.nabla_g(qt_np))
            #print('nabla g', dg.shape)
            loss = alpha1 * F.smooth_l1_loss(p, dg0) + alpha2 * F.smooth_l1_loss(pt, dg)

            ## Second part of loss function: beta1 * (h(q, p) - ((p, f(q, u)) + L(q, u)))
            # Calculate optimal u = -p^T f_u(q, u) (based on adjoint)
            u = (1.0/control_coef)*np.einsum('ijk,ij->ik', env.f_u(q_np), -p_np)
            #print('u', u.shape)
            if dynamic_hidden:
                # (p, f(q, u)) + L(q, u) = (p, qdot_np) + L(q, u)
                qp_dot = hd_net(0, qp)
                qdot, _ = torch.chunk(qp_dot, 2, dim=1)
                #print('qdot, pdot', qdot.shape, pdot.shape)
                qdot_np = qdot.detach().numpy()
                # Calculate reference reduced Hamiltonian using usual Hamiltonian but with supposedly optimal control
                h_pq_ref = np.einsum('ik,ik->i', p_np, qdot_np) + env.L(q_np, u)
            else:
                h_pq_ref = np.einsum('ik,ik->i', p_np, env.f(q_np, u)) + env.L(q_np, u)
            #print('h_pq_ref', h_pq_ref.shape)
            h_pq = hnet(qp)
            #print('h_pq', h_pq.shape)
            loss += beta * F.smooth_l1_loss(h_pq, torch.tensor(h_pq_ref.reshape(-1, 1)))

            # Optimize step
            loss.backward()
            optim.step(); optim.zero_grad()
            # Print progress
            total_loss += loss.item()
            cnt += 1
            if (j+1) % log_interval == 0:
                print('Average loss for {}th iteration is: {}'.format(j+1, total_loss/cnt))
                total_loss = 0
                cnt = 0

# Train phase 2
def train_phase_2(adj_net, hnet, hnet_decoder, z_decoder, z_encoder, qs, T2=1.0, beta=1.0, 
                  num_epoch=20, num_iter=20, batch_size=32, lr=1e-3, log_interval=50):
    
    hd_vae_net = HDVAE(adj_net, hnet, hnet_decoder, z_encoder, z_decoder, T2)
    # Optimizer for Hamiltonian net decoder, (additional) latent encoder and decoder
    optim = torch.optim.Adam(list(hnet_decoder.parameters()) + 
                             list(z_encoder.parameters()) +
                             list(z_decoder.parameters()), lr=lr)
    optim.zero_grad()
    
    # Training over the same data qs num_epoch epochs
    num_samples = qs.shape[0]
    for i in range(num_epoch):
        print('\nEpoch {}: '.format(i+1))
        loss = 0; cnt = 0
        q_dat = torch.clone(qs)[torch.randperm(num_samples)]
        total_loss = 0
        for j in range(num_iter):
            # state training examples
            q = q_dat[np.random.choice(num_samples, batch_size, replace=False)] #[j*batch_size:(j+1)*batch_size]
            # Hamiltonian VAE net returns starting coupled state (state+adjoint)
            # terminal coupled state and its construction, starting state construction
            # mean and logvar of the actual latent variable mapped from terminal state 
            qp, qp_hat, qpt, qpt_hat, mu, logvar = hd_vae_net(q)
            # Reconstruction loss
            loss = F.smooth_l1_loss(qp, qp_hat) +  F.smooth_l1_loss(qpt, qpt_hat)
            # KL loss
            loss += beta * kl_loss(mu, logvar)

            # Optimize step
            loss.backward()
            optim.step(); optim.zero_grad()
            # Print progress
            total_loss += loss.item()
            cnt += 1
            if (j+1) % log_interval == 0:
                print('Average loss for {}th iteration is: {}'.format(j+1, total_loss/cnt))
                total_loss = 0
                cnt = 0

# Main training including phase 1 and phase 2 sequentially
def main_training(env, env_name, qs, 
    adj_net, hnet, hnet_decoder, z_decoder, z_encoder, 
    T1=1, T2=1, control_coef=0.5, dynamic_hidden=False,
    alpha1=1, alpha2=0.1, beta1=1, beta2=1,
    num_epoch1=20, num_epoch2=20, num_iter1=20, num_iter2=20,
    batch_size1=32, batch_size2=32, lr1=1e-3, lr2=1e-3,
    log_interval1=50, log_interval2=50,
    mode=0, retrain_phase1=True, retrain_phase2=True,
    num_examples_phase2=0.2, num_additional_train=5):
    """
    PMP training procedure with different types of modes
    Args:
        adj_net, hnet, hnet_decoder, z_decoder, z_encoder: networks to be traineed
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

    # Train phase 1 only for deterministic Hamiltonian. NeuralPMP-phase1
    if retrain_phase1:
        print('\nTraining phase 1...')
        train_phase_1(env, adj_net, hnet, qs, 
            T1, control_coef, dynamic_hidden, alpha1, alpha2, beta1,
            num_epoch1, num_iter1, batch_size1, lr1, log_interval1)
    else:
        utils.load_models_phase1(adj_net, hnet, env_name)
        print('\nLoaded phase 1 trained models (adjoint net and Hamiltonian net).\n')
    
    # Train both phases: NeuralPMP.
    if mode >= 1:
        if retrain_phase2:
            print('\nTraining phase 2...')
            num_examples = int(num_examples_phase2*qs.shape[0])
            qs2 = torch.clone(qs)[torch.randperm(qs.shape[0])][:num_examples]
            train_phase_2(adj_net, hnet, hnet_decoder, z_decoder, z_encoder, qs2, 
                      T2, beta2, num_epoch2, num_iter2, batch_size2, lr2, log_interval2)
        else:
            utils.load_models_phase2(hnet_decoder, z_encoder, z_decoder, env_name)
            print('\nLoaded phase 2 trained models (Hamiltonian decoder and latent encoder and decoder).\n')
    
    # Use extreme samples (Currently not used).
    if mode == 2:
        qs_extreme_np = get_extreme_samples(env, adj_net, hnet, hnet_decoder, z_decoder, z_encoder, 
                        qs, T=1, num_samples_per_seed=50)
        qs_extreme = torch.tensor(qs_extreme_np, dtype=torch.float)
        
        print('\nRetraining phase 1 with new data...')
        for i in range(num_additional_train):
            print(f'\nRetraining phase 1 with new data {i}th time...')
            # Training using extreme data
            train_phase_1(env, adj_net, hnet, qs_extreme, 
                T1, control_coef, dynamic_hidden, alpha1, alpha2, beta1,
                num_epoch1, num_iter1, batch_size1, lr1, log_interval1)
            # Training using usual data
            train_phase_1(env, adj_net, hnet, qs, 
                T1, control_coef, dynamic_hidden, alpha1, alpha2, beta1,
                num_epoch1, num_iter1, batch_size1, lr1, log_interval1)

    # Save models
    print('\nDone training. Saving model...')
    utils.save_models_phase1(adj_net, hnet, env_name)
    if mode >= 1:
        utils.save_models_phase2(hnet_decoder, z_encoder, z_decoder, env_name)
    print('Done.')

# Convenient single train fct.
def train(env_name, num_examples, mode=0, 
              retrain_phase1=True, retrain_phase2=True,
              num_additional_train=5, num_examples_phase2=0.2,
              arch_file='models/architectures.csv', param_file='models/train_params.csv'):
    print(f'\nTraining environment {env_name}:\n')
    env = utils.get_environment(env_name)
    _, adj_net, hnet, hnet_decoder, z_encoder, z_decoder = utils.get_architectures(arch_file, env_name)
    T1, T2, control_coef, dynamic_hidden, alpha1, alpha2, beta1, beta2, \
        num_epoch1, num_iter1, batch_size1, lr1, log_interval1,\
        num_epoch2, num_iter2, batch_size2, lr2, log_interval2 = utils.get_train_params(param_file, env_name)
    T1, T2 = float(T1), float(T2)

    # Starting point samples
    q_samples = torch.tensor(env.sample_q(num_examples, mode='train'), dtype=torch.float)

    # Main training
    main_training(env, env_name, q_samples, 
        adj_net, hnet, hnet_decoder, z_decoder, z_encoder, 
        T1=T1, T2=T2, control_coef=control_coef, dynamic_hidden=dynamic_hidden,
        alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2,
        num_epoch1=num_epoch1, num_iter1=num_iter1, batch_size1=batch_size1, lr1=lr1, log_interval1=log_interval1, 
        num_epoch2=num_epoch2, num_iter2=num_iter2, batch_size2=batch_size2, lr2=lr2, log_interval2=log_interval2,
        mode=mode, retrain_phase1=retrain_phase1, retrain_phase2=retrain_phase2,
        num_examples_phase2=num_examples_phase2, num_additional_train=num_additional_train)