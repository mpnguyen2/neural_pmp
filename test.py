import numpy as np

import torch
from torchdiffeq import odeint_adjoint as odeint

from common.common_nets import Mlp
from model_nets import HDNet
from envs.classical_controls import ContinuousEnv, MountainCar, Pendulum, CartPole


def test(env, AdjointNet, Hnet, model_name, log_interval=2e3):
    AdjointNet.load_state_dict(torch.load('models/adjoint_' + model_name + '.pth'))
    Hnet.load_state_dict(torch.load('models/hamiltonian_dynamics_' + model_name + '.pth'))
    HDnet = HDNet(Hnet=Hnet)
    q = torch.tensor(env.sample_q(1), dtype=torch.float)
    p = AdjointNet(q)
    qp = torch.cat((q, p), axis=1)
    time_steps = list(np.arange(0, 1, 1e-5))
    traj = odeint(HDnet, qp, torch.tensor(time_steps, requires_grad=True))
    
    cnt = 0
    for e in traj:
        qe, _ = torch.chunk(e, 2, dim=1)
        qe_np = qe.detach().numpy()
        if cnt % log_interval == 0:
            print('Current g-value: {}'.format(env.g(qe_np)[0]))
        cnt += 1


test_mt, test_pendulum, test_cart = False, False, True

if test_mt:
    #Test: Mountain Car
    q_dim = 2; u_dim = 1
    env = MountainCar()
    # Net architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
    
    test(env, adj_net, h_net, model_name='mountain_car')

if test_pendulum:
    #Test: Pendulum
    q_dim = 2; u_dim = 1
    env = Pendulum()
    # Net architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[8, 16, 32])
    
    test(env, adj_net, h_net, model_name='pendulum')
    
if test_cart:
    #Test: Cartpole
    q_dim = 4; u_dim = 1
    env = CartPole()
    # Net architectures
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[16, 32, 32])
    h_net = Mlp(input_dim = 2*q_dim, output_dim=1, layer_dims=[16, 32, 64, 8])
    
    test(env, adj_net, h_net, model_name='cartpole')
    
'''
def test(h_layer_dims, model_path='models/hd.pth', out_file='videos/test8.wmv', 
         num_step=10, log_interval=1000):
    
# Setup fixed knots and grids
    xk, yk = np.mgrid[-1:1:4j, -1:1:4j]
    xg, yg = np.mgrid[-1:1:50j, -1:1:50j]
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    out = cv2.VideoWriter(out_file, fourcc, 20.0, (xg.shape[0], yg.shape[1]), isColor=False)
    
    # Load model
    Hnet = Mlp(input_dim=32, output_dim = 1, layer_dims=h_layer_dims) 
    Hnet.load_state_dict(torch.load(model_path))
    HDnet = HDNet(Hnet=Hnet)
    # HDnet calculate the Hamiltonian dynamics network given the Hamiltonian network Hnet
    qp_one = generate_coords(num_samples=1)
    end = 20.0
    time_steps = list(np.arange(0, end, end/num_step))
    traj = odeint(HDnet, qp_one, torch.tensor(time_steps, requires_grad=True))
    print('Done finding trajectories. Creating videos...')
    for i in range(num_step):
        if (i+1)%log_interval == 1:  
            img = spline_interp(xk, yk, (traj[i][0, :16].detach().numpy()).reshape(xk.shape[0], yk.shape[0]), xg, yg)
            print('Reward in step {} is {}'.format(i//log_interval + 1, isoperi_reward_from_img(img)))
            #print('Adjoint: {}'.format(torch.sum(traj[i][0, 16:]**2)))
            out.write(img)
    
    # Release video
    out.release()
'''