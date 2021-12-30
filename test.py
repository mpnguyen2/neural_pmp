import argparse
import numpy as np
import cv2 

import torch
from torchdiffeq import odeint_adjoint as odeint

from common.common_nets import Mlp
from model_nets import HDNet
from envs.classical_controls import MountainCar, Pendulum, CartPole
from envs.density_optimization import DensityOpt

from train_utils import get_environment, get_architectures

def run_traj(env, AdjointNet, Hnet, env_name, test_trained=True, out_dir='output/optimal_traj_numpy/',
             T=5.0, n_timesteps=50, log_interval=1):
    # Load models
    if test_trained:
        AdjointNet.load_state_dict(torch.load('models/' + env_name + '/adjoint.pth'))
        Hnet.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_dynamics.pth'))
    HDnet = HDNet(Hnet=Hnet)
    # Run optimal trajectory
    q = torch.tensor(env.sample_q(1, mode='test'), dtype=torch.float)
    p = AdjointNet(q)
    qp = torch.cat((q, p), axis=1)
    time_steps = list(np.linspace(0, T+ 1e-5, n_timesteps))
    traj = odeint(HDnet, qp, torch.tensor(time_steps, requires_grad=True))
    print('Done finding trajectory...')
    # Print trajectory and save states to qs
    cnt = 0; eps = 1e-5
    qs = np.zeros((len(traj), q.shape[1]))
    for e in traj:
        qe, pe = torch.chunk(e, 2, dim=1)
        qe_np = qe.detach().numpy()
        qs[cnt, :] = qe_np
        if cnt % log_interval == 0:
            # Print info
            cost = env.g(qe_np)[0]
            #total_energy += env.get_energy(qe_np, pe_np)
            print('step {}: terminal cost {:.3f}'.format(cnt+1, cost))
            if cost < eps:
                break
        cnt += 1
    env.close()
    # Save numpy to out_file
    out_file = out_dir + env_name + '.npy'
    np.save(out_file, qs, allow_pickle=False)

def test(env_name, test_trained=True, T=5.0, n_timesteps=50, log_interval=1):
    # Initialize models and environments
    _, adj_net, hnet, _ =  get_architectures(arch_file='models/architectures.csv', env_name=env_name)
    env = get_environment(env_name) 
    
    # Run trajectory. This use HD models if test_trained is True
    run_traj(env, adj_net, hnet, env_name=env_name, 
                 test_trained=test_trained, 
                 T=T, n_timesteps=n_timesteps, log_interval=log_interval)

def display(env_name, test_trained=True, input_dir='output/optimal_traj_numpy/', out_dir='output/videos/'):
    # Initialize environment
    env = get_environment(env_name) 
    isColor = True
    if env_name == 'shape_opt':
        isColor = False
    if env_name != 'shape_opt':
        env.render(np.zeros(env.q_dim))
    # Initialize video writer
    video_file = out_dir + env_name +'.wmv'
    if not test_trained:
        print('\nTest untrained ' + env_name + ':')
        video_file = out_dir + env_name +'_untrained.wmv'
    else:
        print('\nDisplaying optimal trajectory for ' + env_name + ':')
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    out = cv2.VideoWriter(video_file, fourcc, 20.0, (env.viewer.width, env.viewer.height), isColor=isColor)
    # Load numpy file
    input_file = input_dir+env_name+'.npy'
    qs = np.load(input_file)
    # Write rendering image
    for i in range(qs.shape[0]):
        out.write(env.render(qs[i].reshape(-1)))
    # Release video
    out.release()
    env.close()
    print('\nDone displaying the optimal trajectory!')
    
if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='CLI argument for testing')
    parser.add_argument('env_name', help='Environment to train neural pmp on')
    parser.add_argument('--T', type=int, default=1, help='Terminal time')
    parser.add_argument('--nt', type=int, default=100, help='Number of time steps')
    parser.add_argument('--log', type=int, default=1, help='Log interval')
    parser.add_argument('--display', type=bool, default=False, help='Whether to display (optimal) trajectory')
    args = parser.parse_args()
    # Call train environment
    test(env_name=args.env_name, T=args.T, n_timesteps=args.nt, log_interval=args.log)
    if args.display:
        display(args.env_name)