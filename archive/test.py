import numpy as np
import cv2 
import wandb

import torch
from torchdiffeq import odeint_adjoint as odeint

from common.common_nets import Mlp
from model_nets import HDNet
from envs.classical_controls import MountainCar, Pendulum, CartPole
from envs.density_optimization import DensityOpt

def run_traj(env, AdjointNet, Hnet, model_name, time_steps=list(np.arange(0, 1, 0.1)),
             out_video='videos/test.wmv', test_trained=True, phase2=False, log_interval=1, 
             env_close=True, isColor=True):
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    out = cv2.VideoWriter(out_video, fourcc, 20.0, (env.viewer.width, env.viewer.height), isColor=isColor)
    
    # Load models
    if test_trained:
        AdjointNet.load_state_dict(torch.load('models/' + model_name + '/adjoint.pth'))
        if phase2:
            Hnet.load_state_dict(torch.load('models/' + model_name + '/hamiltonian_decoder.pth'))
        else:
            Hnet.load_state_dict(torch.load('models/' + model_name + '/hamiltonian_dynamics.pth'))
    # Build symplectic dynamics net from Hamiltonian net
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
            wandb.log({"Terminal cost": env.g(qe_np)[0]})
            # Write rendering image
            out.write(env.render(qe_np.reshape(-1)))
        cnt += 1

    # Release video
    out.release()
    
    if env_close and model_name != 'shape_opt':
        env.close()
    
def next_state(q, AdjointNet, HDnet, time_step=1):
    p = AdjointNet(q)
    qp = torch.cat((q, p), axis=1)
    next_qp = odeint(HDnet, qp, torch.tensor([0, time_step], requires_grad=False))[-1]
    print(torch.sum((next_qp-qp)**2))
    next_q, _ = torch.chunk(next_qp, 2, dim=1)

    return next_q

# Store all network architectures for all environments
adj_nets = {}
h_nets = {}
q_dims = {}
# Mountain car
q_dim=2; q_dims['mountain_car'] = q_dim
adj_nets['mountain_car'] = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
h_nets['mountain_car'] = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
# Cartpole
q_dim=4; q_dims['cartpole'] = q_dim
adj_nets['cartpole'] = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[16, 32, 32])
h_nets['cartpole'] = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[16, 32, 64, 8])
# Pendulum
q_dim=2; q_dims['pendulum'] = q_dim
adj_nets['pendulum'] = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
h_nets['pendulum'] = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
# Shape opt
q_dim = 16; q_dims['shape_opt'] = q_dim
adj_nets['shape_opt'] = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[32, 64])
h_nets['shape_opt'] = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[64, 8])

def test(model_name, test_trained=True, phase2=False,
         time_steps=list(np.arange(0, 1, 0.1)), log_interval=1):
    # Initialize environment
    isColor = True
    if model_name == 'mountain_car':
        env = MountainCar()
    if model_name == 'cartpole':
        env = CartPole()
    if model_name == 'pendulum':
        env = Pendulum()
    if model_name == 'shape_opt':
        env = DensityOpt()
        isColor = False
    if model_name != 'shape_opt':
        env.render(np.zeros(q_dims[model_name]))
    # Initialize video path
    video_path = 'videos/test_'+ model_name +'.wmv'
    if not test_trained:
        print('\nTest untrained ' + model_name + ':')
        video_path = 'videos/test_'+ model_name +'_untrained.wmv'
    elif phase2:
        print('Test phase 2 for ' + model_name + ':')
        video_path = 'videos/test_'+ model_name +'_phase2.wmv'
    else:
        print('Test ' + model_name + ':')
    # Run trajectory. This use HD models if test_trained is True
    run_traj(env, adj_nets[model_name], h_nets[model_name], model_name=model_name, 
                 test_trained=test_trained, phase2=phase2,
                 time_steps=time_steps, log_interval=log_interval,
                 out_video=video_path, isColor=isColor)
    
test_mt, test_cart, test_pendulum, test_density = False, False, True, False
if test_mt:
    test('mountain_car', time_steps=list(np.arange(0, 130, 0.4)), log_interval=1, test_trained=True)
if test_cart:
    test('cartpole', time_steps=list(np.arange(0, 200, 0.4)), log_interval=2, test_trained=True)
if test_pendulum:
    test('pendulum', time_steps=list(np.arange(0, 1000, 1.0)), log_interval=1, test_trained=True)
if test_density:
    test('shape_opt', time_steps=list(np.arange(0, 100, 0.4)), log_interval=1, test_trained=True)
    
test_mt2, test_cart2, test_pendulum2, test_density2 = False, False, False, False
if test_mt2:
    test('mountain_car', time_steps=list(np.arange(0, 24, 0.05)), log_interval=1, test_trained=True, phase2=True)
if test_cart2:
    test('cartpole', time_steps=list(np.arange(0, 100, 0.1)), log_interval=1, test_trained=True, phase2=True)
if test_pendulum2:
    test('pendulum', test_trained=True, phase2=True)
if test_density2:
    test('shape_opt', time_steps=list(np.arange(0, 100, 0.4)), log_interval=10, test_trained=True, phase2=True)
 