import argparse, time
import numpy as np
import cv2 

import torch
from torchdiffeq import odeint_adjoint as odeint

#from common.common_nets import Mlp
from model_nets import HDNet
#from envs.classical_controls import MountainCar, Pendulum, CartPole
from envs.density_optimization import DensityOpt

from train_utils import get_environment, get_architectures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_traj(env, adj_net, hnet, env_name, use_adj=False, use_hnet=True, out_dir='output/optimal_traj_numpy/',
             T=5.0, n_timesteps=50, log_interval=1):
    # Load models
    if use_hnet:
        hnet.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_dynamics.pth'))
    HDnet = HDNet(Hnet=hnet).to(device)
    if use_adj:
        adj_net.load_state_dict(torch.load('models/' + env_name + '/adjoint.pth'))
        adj_net = adj_net.to(device)
    # Sample state
    q = torch.tensor(env.sample_q(1, mode='test'), dtype=torch.float, device=device)
    print('\nInitial cost: {:.3f}.'.format(env.g(q.cpu().detach().numpy())[0]))
    nabla_s = env.nabla_g(q.cpu().detach().numpy())
    # Finding appropriate adjoint variable either by adjoint network
    # or by searching randomly over the one give the best terminal state given the fixed and trained hamiltonian dynamics
    if use_adj:
        p = adj_net(q)
        #print(p.cpu().detach().numpy())
    else:
        start_time = time.time()
        num_trials = 500
        q_dup = torch.cat([q for _ in range(num_trials)])
        ps = torch.rand(num_trials, q.shape[1], dtype=torch.float, device=device) - .5 #AdjointNet(q)
        min_ind = -1; min_val = 1e9
        test_times = list(np.linspace(0, T, 2))
        test_times = torch.tensor(test_times, requires_grad=True, device=device)
        print('Looking for optimal starting adjoint...')
        qp = torch.cat((q_dup, ps), axis=1)
        with torch.no_grad():
            traj = odeint(HDnet, qp, test_times)
        print('Done running batch of testing adjoints p and duplicate of states q')
        norms = []
        for i in range(num_trials):
            qt, pt = torch.chunk(traj[-1, i:(i+1)], 2, dim=1)
            #cost = np.sum(pt.cpu().detach().numpy()**2)
            #print(cost)
            cost = env.g(qt.cpu().detach().numpy())[0]
            norms.append(np.sum(pt.cpu().detach().numpy()**2))
            if cost < min_val:
                min_ind = i
                min_val = cost
        p = ps[min_ind:(min_ind+1)]
        print('Finding optimal p takes {:.4f} seconds.'.format(time.time()-start_time))
        norms = np.array(norms)
        #print(norms[min_ind])
        #print(np.min(norms), np.max(norms), np.sort(norms)[:10])

    print('Done finding optimal starting adjoint. Finding optimal trajectory...')

    # Given p, run the (optimal) trajectory
    cnt = 0; eps = 1e-5
    qp = torch.cat((q, p), axis=1)
    time_steps = list(np.linspace(0, T+ 1e-5, n_timesteps))
    time_steps = torch.tensor(time_steps, requires_grad=True, device=device)
    with torch.no_grad():
        traj = odeint(HDnet, qp, time_steps)
    print('Done finding trajectory...')
    # Then save states on the trajectory to qs
    qs = np.zeros((len(traj), q.shape[1]))
    print(log_interval)

    for e in traj:
        qe, _ = torch.chunk(e, 2, dim=1)
        qe_np = qe.cpu().detach().numpy()
        qs[cnt, :] = qe_np
        if cnt % log_interval == 0:
            # Print info
            cost = env.g(qe_np)[0]
            #total_energy += env.get_energy(qe_np, pe_np)
            print('step {}: terminal cost {:.3f}'.format(cnt+1, cost))
            #print('q:', qe.reshape(-1)[0])
            #if cost < eps:
                #break
        cnt += 1
    # Print numerical information
    nabla_t = env.nabla_g(qs[-1:])
    _, pt = torch.chunk(traj[-1], 2, dim=1)
    print('\nSome numerical information for further debugging:')
    print('Starting nabla:', nabla_s)
    print('Terminal nabla:', nabla_t)
    print('Adjoint:', p.cpu().detach().numpy())
    print('Terminal adjoint:', pt.cpu().detach().numpy())

    env.close()

    # Save numpy to out_file
    out_file = out_dir + env_name + '.npy'
    np.save(out_file, qs, allow_pickle=False)

def test(env_name, use_adj=False, use_hnet=True, T=5.0, n_timesteps=50, log_interval=1):
    # Initialize models and environments
    _, adj_net, hnet, _ =  get_architectures(arch_file='models/architectures.csv', env_name=env_name)
    env = get_environment(env_name) 
    
    # Run trajectory. This use HD models if test_trained is True
    run_traj(env, adj_net, hnet, env_name=env_name, 
                 use_adj=use_adj, use_hnet=use_hnet,
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
    parser.add_argument('--use_adj', type=bool, default=False, help='Whether to use adjacency network or greedily select the best adjoint state')
    parser.add_argument('--T', type=float, default=1, help='Terminal time')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of time steps')
    parser.add_argument('--log', type=int, default=1, help='Log interval')
    parser.add_argument('--run_traj', type=bool, default=False, help='Whether to run trajectory and save it to np file')
    parser.add_argument('--display', type=bool, default=False, help='Whether to display (optimal) trajectory')
    args = parser.parse_args()
    # Call train environment
    if args.run_traj:
        test(env_name=args.env_name, use_adj=args.use_adj, T=args.T, n_timesteps=args.num_steps, log_interval=args.log)
    if args.display:
        display(args.env_name)