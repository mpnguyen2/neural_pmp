import numpy as np; np.random.seed(1)
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import torch
from envs.classical_controls import MountainCar, CartPole, Pendulum
from common_nets import Mlp, Encoder

def toList(s):
    tokens = s[1:-1].split(", ")
    ans = []
    for token in tokens:
        ans.append(int(token))
    return ans

# Get environment
def get_environment(env_name):
    if env_name == 'mountain_car':
        return MountainCar()
    if env_name == 'cartpole':
        return CartPole()
    if env_name == 'pendulum':
        return Pendulum()

# Get architecture
def get_architectures(arch_file, env_name):
    # Get architecture info from arch_file
    df = pd.read_csv(arch_file)
    info = df[df['env_name']==env_name]
    # Extract architecture info
    q_dim=info['q_dim'].values[0]
    adj_net_layer_dims = toList(info['adj_net_layer_dims'].values[0])
    hnet_layer_dims = toList(info['hnet_layer_dims'].values[0])
    hnet_decoder_layer_dims = toList(info['hnet_decoder_layer_dims'].values[0])
    z_dim = int(info['z_dim'].values[0])
    z_encoder_share_layer_dims = toList(info['z_encoder_share_layer_dims'].values[0])
    z_encoder_mean_layer_dims = toList(info['z_encoder_mean_layer_dims'].values[0])
    z_encoder_logvar_layer_dims = toList(info['z_encoder_logvar_layer_dims'].values[0])
    z_decoder_layer_dims = toList(info['z_decoder_layer_dims'].values[0])
    # Build adjoint variable and Hamiltonian nets
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=adj_net_layer_dims)
    hnet = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=hnet_layer_dims)
    # Build hnet_decoder
    hnet_decoder = Mlp(input_dim=2*q_dim, output_dim=1, layer_dims=hnet_decoder_layer_dims)
    # Build latent z_encoder
    z_encoder = Encoder(input_dim=2*q_dim, share_layer_dims=z_encoder_share_layer_dims, 
        mean_layer_dims=z_encoder_mean_layer_dims, 
        logvar_layer_dims=z_encoder_logvar_layer_dims, 
        output_dim=z_dim)
    # Build latent z_decoder
    z_decoder = Mlp(input_dim=z_dim, output_dim=2*q_dim, layer_dims=z_decoder_layer_dims)
    
    return q_dim, adj_net, hnet, hnet_decoder, z_encoder, z_decoder

# Get training (hyper)parameters.
def get_train_params(param_file, env_name):
    # Get parameter info from param file
    df = pd.read_csv(param_file)
    info = df[df['env_name']==env_name]
    # Get terminal times T1, T2
    T1, T2 = info['T1'].values[0], info['T2'].values[0]
    # Get dynamic_hiden options and control coefficients 
    control_coef, dynamic_hidden = info['control_coef'].values[0], info['dynamic_hidden'].values[0]
    # Get hyperparameter loss function 
    alpha1, alpha2, beta1, beta2 = info['alpha1'].values[0], info['alpha2'].values[0], info['beta1'].values[0], info['beta2'].values[0]
    # Get training details for first phase (batch_size, num_epoch, lr, log_interval)
    num_epoch1, num_iter1, batch_size1, lr1, log_interval1 = info['num_epoch1'].values[0], info['num_iter1'].values[0],\
        info['batch_size1'].values[0],info['lr1'].values[0], info['log_interval1'].values[0]
    # Get training details for second phase (batch_size, num_epoch, lr, log_interval)
    num_epoch2, num_iter2, batch_size2, lr2, log_interval2 = info['num_epoch2'].values[0], info['num_iter2'].values[0],\
        info['batch_size2'].values[0], info['lr2'].values[0], info['log_interval2'].values[0]

    return T1, T2, control_coef, dynamic_hidden, alpha1, alpha2, beta1, beta2,\
        num_epoch1, num_iter1, batch_size1, lr1, log_interval1,\
        num_epoch2, num_iter2, batch_size2, lr2, log_interval2

# save model phase 1
def save_models_phase1(adj_net, hnet, env_name):
    torch.save(adj_net.state_dict(), 'models/' + env_name + '/adjoint.pth')
    torch.save(hnet.state_dict(), 'models/' + env_name + '/hamiltonian_dynamics.pth')

# load model phase 1
def load_models_phase1(adj_net, hnet, env_name):
    adj_net.load_state_dict(torch.load('models/' + env_name + '/adjoint.pth'))
    hnet.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_dynamics.pth'))
    
# save model phase 2
def save_models_phase2(hnet_decoder, z_encoder, z_decoder, env_name):
    torch.save(hnet_decoder.state_dict(), 'models/' + env_name + '/hamiltonian_decoder.pth')
    torch.save(z_encoder.state_dict(), 'models/' + env_name + '/z_encoder.pth')
    torch.save(z_decoder.state_dict(), 'models/' + env_name + '/z_decoder.pth')
    
# load model phase 1
def load_models_phase2(hnet_decoder, z_encoder, z_decoder, env_name):
    hnet_decoder.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_decoder.pth'))
    z_encoder.load_state_dict(torch.load('models/' + env_name + '/z_encoder.pth'))
    z_decoder.load_state_dict(torch.load('models/' + env_name + '/z_decoder.pth'))

### Plotting
def _bootstrap(data, n_boot=2000, ci=68):
    boot_dist = []
    for _ in range(int(n_boot)):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(np.mean(sample, axis=0))
    b = np.array(boot_dist)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
    return (s1,s2)

def _tsplot(ax, x, data, mode='bootstrap', **kw):
    est = np.mean(data, axis=0)
    if mode == 'bootstrap':
        cis = _bootstrap(data)
    else:
        sd = np.std(data, axis=0)
        cis = (est - sd, est + sd)
    p2 = ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    p1 = ax.plot(x, est, **kw)
    ax.margins(x=0)

    return p1, p2

def plot_eval_benchmarks(eval_dict, time_steps, title, mode='bootstrap', 
                         colors=['red', 'blue', 'green', 'orange'],
                         plot_dir='tmp.png'):
    methods = list(eval_dict.keys())
    ax = plt.gca()
    graphic_list = []
    for i, method in enumerate(methods):
        data = eval_dict[method]
        _, p2 = _tsplot(ax, np.array(time_steps), data, mode, label=method, color=colors[i])
        graphic_list.append(p2)
    ax.legend(graphic_list, methods)
    ax.set_title(title)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Evaluation cost')
    plt.savefig('plots/' + plot_dir)
    plt.show()