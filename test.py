import numpy as np
import cv2 
import torch
from torchdiffeq import odeint_adjoint as odeint
from model_nets import HDNet
import utils

def run_traj(env, adj_net, hnet, hnet_decoder, env_name,
             num_trajs, time_steps, test_trained, phase2,
             save_video=False, video_path='videos/test.wmv',
             total_random=False):
    
    if save_video:
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'WMV1')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (env.screen_width, env.screen_height), isColor=True)
        
    # Load models
    if test_trained:
        adj_net.load_state_dict(torch.load('models/' + env_name + '/adjoint.pth'))
        if phase2:
            hnet_decoder.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_decoder.pth'))
        else:
            hnet.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_dynamics.pth'))

    # Build symplectic dynamics net from Hamiltonian net from phase 1 or Hamiltonian decoder from phase 2.
    if phase2:
        HDnet = HDNet(hnet=hnet_decoder)
    else:
        HDnet = HDNet(hnet=hnet)
    
    if total_random:
        final_costs = []
        start_q = env.sample_q(num_trajs, mode='test')
        num_steps = len(time_steps)
        for t in range(num_trajs):
            cur_traj_cost = []
            q = start_q[t]
            for _ in range(num_steps):
                cur_traj_cost.append(env.eval(q.reshape(1, -1))[0])
                a = 0.3*q*(np.random.rand(q.shape[0])-0.5)
                q = q + a
            final_costs.append(np.array(cur_traj_cost))
        return np.array(final_costs)
    
    # Run optimal trajectory
    q = torch.tensor(env.sample_q(num_trajs, mode='test'), dtype=torch.float)
    p = adj_net(q)
    qp = torch.cat((q, p), axis=1)
    traj = odeint(HDnet, qp, torch.tensor(time_steps, requires_grad=False))
    print('Done finding trajectory...')

    # Collect results and optionally save to videos
    final_costs = []
    qe_np_all = []
    for e in traj:
        qe, _ = torch.chunk(e, 2, dim=1)
        qe_np = qe.detach().numpy()
        qe_np_all.append(qe_np[0])
        #qe_np_all.append(qe_np.reshape(-1))
        final_costs.append(env.eval(qe_np))
        if save_video:
            # Write rendering image
            out.write(env.render(qe_np.reshape(-1)))
    #print(qe_np_all)
    #print(np.array(final_costs).shape)

    # Release video
    if save_video:
        out.release()

    env.close()
    # (num_traj, num_step)
    return np.swapaxes(np.array(final_costs, dtype=float), 0, 1)

def _test(env_name, num_trajs, time_steps, test_trained, phase2, total_random=False):
    # Initialize models (this first to take state dimension q_dim)
    _, adj_net, hnet, hnet_decoder, _, _ = \
        utils.get_architectures(arch_file='models/architectures.csv', env_name=env_name)
    env = utils.get_environment(env_name)

    return run_traj(env, adj_net, hnet, hnet_decoder, env_name,
                    num_trajs, time_steps,
                    test_trained, phase2,
                    total_random=total_random)

def benchmarks(env_name, num_trajs, time_steps,
               eval_along_traj=False,
               plot_traj=False,
               y_label='value',
               y_label_traj='functional'):
    # Calculate result
    final_costs_total_random = _test(env_name, num_trajs, time_steps, test_trained=False,
                                     phase2=False, total_random=True)
    final_costs_untrained = _test(env_name, num_trajs, time_steps, test_trained=False, phase2=False)
    final_costs_phase_1 = _test(env_name, num_trajs, time_steps, test_trained=True, phase2=False)
    final_costs_phase_2 = _test(env_name, num_trajs, time_steps, test_trained=True, phase2=True)

    # Draw (statistical) plot
    eval_dict = {
        'Random': final_costs_total_random,
        'Random Hamiltonian': final_costs_untrained,
        'NeuralPMP-phase 1': final_costs_phase_1,
        'NeuralPMP': final_costs_phase_2
    }
    utils.plot_eval_benchmarks(eval_dict, time_steps,
                               title='Benchmarkings on ' + env_name,
                               y_label=y_label,
                               plot_dir=env_name + '_benchmarks_plot.png')

    # Report statistics
    end_cost_total_random = np.mean(final_costs_total_random[:, -1])
    end_cost_untrained = np.mean(final_costs_untrained[:, -1])
    end_cost_phase_1 = np.mean(final_costs_phase_1[:, -1])
    end_cost_phase_2 = np.mean(final_costs_phase_2[:, -1])
    print('Random:', end_cost_total_random)
    print('Random Hamiltonian:', end_cost_untrained)
    print('NeuralPMP-phase 1:', end_cost_phase_1)
    print('NeuralPMP:', end_cost_phase_2)

    # Evaluate along trajectory
    if eval_along_traj:
        env = utils.get_environment(env_name)
        eval_traj_dict = {
            'Random': env.eval_all(final_costs_total_random),
            'Random Hamiltonian': env.eval_all(final_costs_untrained),
            'NeuralPMP-phase 1': env.eval_all(final_costs_phase_1),
            'NeuralPMP': env.eval_all(final_costs_phase_2)
        }
        
        print('\nEvaluation along trajectories:')
        print('Random:', np.mean(eval_traj_dict['Random']))
        print('Random Hamiltonian:', np.mean(eval_traj_dict['Random Hamiltonian']))
        print('NeuralPMP-phase 1:', np.mean(eval_traj_dict['NeuralPMP-phase 1']))
        print('NeuralPMP:', np.mean(eval_traj_dict['NeuralPMP']))
        utils.plot_eval_traj_benchmarks(eval_traj_dict,
                               title='Benchmarkings on ' + env_name,
                               plot_dir=env_name + '_functional_benchmarks_plot.png',
                               y_label=y_label_traj)
    
    # Plot trajectories
    if plot_traj:
        utils.plot_trajs(final_costs_total_random, time_steps, title='Untrained')
        utils.plot_trajs(final_costs_phase_1, time_steps, title='Phase 1')
        utils.plot_trajs(final_costs_phase_2, time_steps, title='Phase 2')
        utils.plot_trajs(final_costs_untrained, time_steps, title='Random Hamiltonian')

def visualize(env_name, time_steps, test_trained, phase2):
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

    # Initialize/load models
    _, adj_net, hnet, hnet_decoder, _, _ = \
        utils.get_architectures(arch_file='models/architectures.csv', env_name=env_name)
    env = utils.get_environment(env_name) 
    run_traj(env, adj_net, hnet, hnet_decoder, env_name,
             1, time_steps, test_trained, phase2,
             save_video=True, video_path=video_path)