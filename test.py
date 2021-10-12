import numpy as np
import cv2 

import torch
from torchdiffeq import odeint_adjoint as odeint

from common.common_nets import Mlp
from model_nets import HDNet
from envs.classical_controls import MountainCar, Pendulum, CartPole


def test(env, AdjointNet, Hnet, model_name, out_video, log_interval=1):
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    out = cv2.VideoWriter(out_video, fourcc, 20.0, (env.viewer.width, env.viewer.height), isColor=True)

    # Load models
    AdjointNet.load_state_dict(torch.load('models/adjoint_' + model_name + '.pth'))
    Hnet.load_state_dict(torch.load('models/hamiltonian_dynamics_' + model_name + '.pth'))
    # Build symplectic dynamics net from Hamiltonian net
    HDnet = HDNet(Hnet=Hnet)
    # Run optimal trajectory
    q = torch.tensor(env.sample_q(1), dtype=torch.float)
    p = AdjointNet(q)
    qp = torch.cat((q, p), axis=1)
    time_steps = list(np.arange(0, 10, 0.01))
    traj = odeint(HDnet, qp, torch.tensor(time_steps, requires_grad=True))
    
    print('Done finding trajectory...')
    # Print trajectory and save images to vid
    cnt = 0
    for e in traj:
        qe, _ = torch.chunk(e, 2, dim=1)
        qe_np = qe.detach().numpy()
        if cnt % log_interval == 0:
            #print('Q:', qe.detach().numpy())
            # Print info
            print('terminal cost value: {}'.format(env.g(qe_np)[0]))
            # Write rendering image
            out.write(env.render(qe_np.reshape(-1)))
        cnt += 1

    # Release video
    out.release()
    
    env.close()

test_mt, test_pendulum, test_cart = False, True, False

if test_mt:
    #Test: Mountain Car
    q_dim = 2; u_dim = 1
    env = MountainCar()
    env.render(np.zeros(q_dim))
    # Net architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
    
    print('\nTest mountain car:')
    test(env, adj_net, h_net, model_name='mountain_car', out_video='videos/test_mountain_car.wmv')

if test_pendulum:
    #Test: Pendulum
    q_dim = 2; u_dim = 1
    env = Pendulum()
    env.render(np.zeros(q_dim))
    # Net architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
    
    print('\nTest pendulum:')
    test(env, adj_net, h_net, model_name='pendulum', out_video='videos/test_pendulum.wmv')
    
if test_cart:
    #Test: Cartpole
    q_dim = 4; u_dim = 1
    env = CartPole()
    env.render(np.zeros(q_dim))
    # Net architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[16, 32, 32])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[16, 32, 64, 8])
    
    print('\nTest cartpole:')
    test(env, adj_net, h_net, model_name='cartpole', out_video='videos/test_cartpole.wmv')
