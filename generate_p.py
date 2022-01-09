import argparse, time
import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint

from model_nets import HDNet

from train_utils import get_environment, get_architectures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_optimal_p(q, env, env_name, T, num_trials=500):
    _, _, hnet, _ =  get_architectures(arch_file='models/architectures.csv', env_name=env_name)
    hnet.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_dynamics.pth'))
    HDnet = HDNet(Hnet=hnet).to(device)
    # Finding appropriate adjoint variable 
    # by searching randomly over the one give the best terminal state given the fixed and trained hamiltonian dynamics
    q_dup = torch.cat([q for _ in range(num_trials)])
    ps = torch.rand(num_trials, q.shape[1], dtype=torch.float, device=device) - .5 #AdjointNet(q)
    min_ind = -1; min_val = 1e9
    test_times = list(np.linspace(0, T, 2))
    test_times = torch.tensor(test_times, requires_grad=True, device=device)
    #print('Looking for optimal starting adjoint...')
    qp = torch.cat((q_dup, ps), axis=1)
    with torch.no_grad():
        traj = odeint(HDnet, qp, test_times)
    for i in range(num_trials):
        qt, _ = torch.chunk(traj[-1, i:(i+1)], 2, dim=1)    
        cost = env.g(qt.cpu().detach().numpy())[0]
        if cost < min_val:
            min_ind = i
            min_val = cost
    p = ps[min_ind:(min_ind+1)].cpu().detach().numpy().reshape(-1)

    return np.concatenate((q.cpu().detach().numpy().reshape(-1), p))

def generate_adjoint_data(num_samples, env_name, T, num_trials, out_dir='output/generated_qp/'):
    start_time = time.time()
    env = get_environment(env_name) 
    result = []
    for i in range(num_samples):
        q = torch.tensor(env.sample_q(1, mode='test'), dtype=torch.float, device=device)
        qp_np = generate_optimal_p(q, env, env_name, T, num_trials)
        print('Generated {} samples.'.format(i+1))
        result.append(qp_np)
    
    # Save result to npy file
    result = np.array(result, dtype=float)
    print(result.shape)
    out_file = out_dir + env_name + '.npy'
    np.save(out_file, result, allow_pickle=False)
    print('Generating data takes {:.3f} seconds.'.format(time.time()-start_time))

if __name__ == '__main__':
    parser =argparse.ArgumentParser(description='CLI argument for generating p data')
    parser.add_argument('env_name', help='Environment to train neural pmp on')
    parser.add_argument('--num_samples', type=int, default=400, help='Number of (q, p) samples to generate')
    parser.add_argument('--T', type=float, default=2, help='Terminal time for each trajectory')
    parser.add_argument('--num_trials', type=int, default=500, help='Number of trial to find optimal p per q sample')
    args = parser.parse_args()
    generate_adjoint_data(args.num_samples, args.env_name, args.T, args.num_trials)
