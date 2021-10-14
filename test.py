import numpy as np
import cv2 

import torch
from torchdiffeq import odeint_adjoint as odeint

from common.common_nets import Mlp
from model_nets import HDNet
from envs.classical_controls import MountainCar, Pendulum, CartPole
from envs.density_optimization import DensityOpt

def run_traj(env, AdjointNet, Hnet, model_name, out_video, log_interval=1, 
         test_trained=True, env_close=True, isColor=True):
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    out = cv2.VideoWriter(out_video, fourcc, 20.0, (env.viewer.width, env.viewer.height), isColor=isColor)
    
    # Load models
    if test_trained:
        AdjointNet.load_state_dict(torch.load('models/' + model_name + '/adjoint.pth'))
        Hnet.load_state_dict(torch.load('models/' + model_name + '/hamiltonian_dynamics.pth'))
    # Build symplectic dynamics net from Hamiltonian net
    HDnet = HDNet(Hnet=Hnet)
    
    # Run optimal trajectory
    q = torch.tensor(env.sample_q(1, mode='test'), dtype=torch.float)
    p = AdjointNet(q)
    qp = torch.cat((q, p), axis=1)
    time_steps = list(np.arange(0, 100, 0.1))
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
adj_nets['shape_opt'] = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[32, 64, 128])
h_nets['shape_opt'] = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[64, 128, 32, 8])

def test(model_name, test_trained=True, numRun=1):
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
    if test_trained:
        print('Test ' + model_name + ':')
        numGoodRun = run_traj(env, adj_nets[model_name], h_nets[model_name], isColor=isColor,
             model_name=model_name, out_video='videos/test_'+ model_name +'.wmv')
    else:
        print('\nTest untrained ' + model_name + ':')
        numGoodRun = run_traj(env, adj_nets[model_name], h_nets[model_name], test_trained=False, isColor=isColor,
            model_name=model_name, out_video='videos/test_'+ model_name +'_untrained.wmv')
        
    return numGoodRun
    
test_mt, test_cart, test_pendulum, test_density = True, False, False, False
if test_mt:
    test('mountain_car', test_trained=True)
if test_cart:
    test('cartpole', test_trained=True)
if test_pendulum:
    test('pendulum', test_trained=True)
if test_density:
    test('shape_opt', test_trained=True)
 