import numpy as np
import cv2 

import torch
from torchdiffeq import odeint_adjoint as odeint

from common.common_nets import Mlp
from model_nets import HDNet
from envs.classical_controls import MountainCar, Pendulum, CartPole
from train_utils import get_environment, get_architectures

def run_traj(env, AdjointNet, Hnet, HnetDecoder, env_name, time_steps=list(np.arange(0, 1, 0.1)),
             out_video='videos/test.wmv', test_trained=True, phase2=False, log_interval=1, 
             env_close=True, isColor=True):
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    out = cv2.VideoWriter(out_video, fourcc, 20.0, (env.viewer.width, env.viewer.height), isColor=isColor)
    
    # Load models
    if test_trained:
        AdjointNet.load_state_dict(torch.load('models/' + env_name + '/adjoint.pth'))
        if phase2:
            HnetDecoder.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_decoder.pth'))
        else:
            Hnet.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_dynamics.pth'))
    # Build symplectic dynamics net from Hamiltonian net
    if phase2:
        HDnet = HDNet(Hnet=HnetDecoder)
    else:
        HDnet = HDNet(Hnet=Hnet)
    
    # Run optimal trajectory
    q = torch.tensor(env.sample_q(1, mode='test'), dtype=torch.float)
    p = AdjointNet(q)
    qp = torch.cat((q, p), axis=1)
    
    traj = odeint(HDnet, qp, torch.tensor(time_steps, requires_grad=False))
    print('Done finding trajectory...')
    # Print trajectory and save images to vid
    cnt = 0
    for e in traj:
        qe, _ = torch.chunk(e, 2, dim=1)
        qe_np = qe.detach().numpy()
        if cnt % log_interval == 0:
            #print('Q:', qe.detach().numpy())
            # Print info
            print('terminal cost: {}'.format(env.g(qe_np)[0]))
            # Write rendering image
            out.write(env.render(qe_np.reshape(-1)))
        cnt += 1

    # Release video
    out.release()
    
    if env_close and env_name != 'shape_opt':
        env.close()
    
def next_state(q, AdjointNet, HDnet, time_step=1):
    p = AdjointNet(q)
    qp = torch.cat((q, p), axis=1)
    next_qp = odeint(HDnet, qp, torch.tensor([0, time_step], requires_grad=False))[-1]
    print(torch.sum((next_qp-qp)**2))
    next_q, _ = torch.chunk(next_qp, 2, dim=1)

    return next_q

def test(env_name, test_trained=True, phase2=False,
         time_steps=list(np.arange(0, 1, 0.1)), log_interval=1):
    
    # Initialize models (this first to take state dimension q_dim)
    q_dim, adj_net, hnet, hnet_decoder, _, _ = \
            get_architectures(arch_file='models/architectures.csv', env_name=env_name)
    # Initialize environment
    isColor = True
    env = get_environment(env_name) 
    if env_name == 'shape_opt':
        isColor = False
    if env_name != 'shape_opt':
        env.render(np.zeros(q_dim))
    # Initialize video path
    video_path = 'videos/test_'+ env_name +'.wmv'
    if not test_trained:
        print('\nTest untrained ' + env_name + ':')
        video_path = 'videos/test_'+ env_name +'_untrained.wmv'
    elif phase2:
        print('Test phase 2 for ' + env_name + ':')
        video_path = 'videos/test_'+ env_name +'_phase2.wmv'
    else:
        print('Test ' + env_name + ':')
    
    # Run trajectory. This use HD models if test_trained is True
    run_traj(env, adj_net, hnet, hnet_decoder, env_name=env_name, 
                 test_trained=test_trained, phase2=phase2,
                 time_steps=time_steps, log_interval=log_interval,
                 out_video=video_path, isColor=isColor)
    
test_mt, test_cart, test_pendulum = False, False, False
if test_mt:
    test('mountain_car', time_steps=list(np.arange(0, 200, 1.0)), log_interval=1, test_trained=True)
if test_cart:
    test('cartpole', time_steps=list(np.arange(0, 200, 0.4)), log_interval=2, test_trained=True)
if test_pendulum:
    test('pendulum', time_steps=list(np.arange(0, 1000, 1.0)), log_interval=1, test_trained=True)
    
test_mt2, test_cart2, test_pendulum2 = False, False, False
if test_mt2:
    test('mountain_car', time_steps=list(np.arange(0, 2, 0.1)), log_interval=1, test_trained=True, phase2=True)
if test_cart2:
    test('cartpole', time_steps=list(np.arange(0, 100, 0.1)), log_interval=1, test_trained=True, phase2=True)
if test_pendulum2:
    test('pendulum', test_trained=True, phase2=True)