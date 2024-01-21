import numpy as np
import torch
import torch.nn as nn
from common_nets import Mlp
import matplotlib.pyplot as plt

device_default = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time
start_time = time.time()

A = 0; B = 1; Q = 1; R = 0
A = torch.tensor(A).to(device_default)
B = torch.tensor(B).to(device_default)
Q = torch.tensor(Q).to(device_default)
R = torch.tensor(R).to(device_default)

def f_x(x, u):
    return A

def l_x(x, u):
    return Q*x

def H_x(x, u, lmd):
    return lmd * f_x(x, u) + l_x(x, u)

def adjoint_blackbox(x_n, u_n, lambda_n, t_n, t_p):
    return lambda_n + (t_n - t_p) * H_x(x_n, u_n, lambda_n) 

def H_u(x, u, lmd):
    return B*lmd + R*u

def dynamic_blackbox(u, t, x0):
    n = u.shape[1]
    x = [x0]*n
    for i in range(n-1):
        x[i+1] = x[i] + (t[i+1]-t[i])*(A*x[i] + B*u[:, i:(i+1)])

    return x

def train(actor_net, t, batch_size=32, lr=1e-3, N_iter=100, device=device_default, log_interval=5):
    optim_hnet = torch.optim.Adam(actor_net.parameters(), lr=lr)
    optim_hnet.zero_grad()
    total_loss = 0
    num_times = len(t)
    losses = []
    for i in range(num_times):
        t[i] = torch.tensor(t[i]).to(device)
    for iter in range(N_iter):
        x0 = (torch.rand(batch_size, 1) - 0.5) * 10
        x0 = x0.to(device)
        u = actor_net(x0)
        x = dynamic_blackbox(u, t, x0)
        lmds = [torch.zeros(1, dtype=float, device=device)]*num_times
        for i in range(num_times-1, 0, -1):
            lmds[i-1] = adjoint_blackbox(x[i], u[:, i], lmds[i], t[i], t[i-1])

        loss = torch.vstack([H_u(x[i], u[:, i], lmds[i]) for i in range(num_times)])
        loss = torch.sum(loss**2)
        # Optimize model
        optim_hnet.zero_grad()
        loss.backward()
        # for param in actor_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        optim_hnet.step()
        total_loss += loss
        
        if iter % log_interval == log_interval-1:
            avg_loss = total_loss.cpu().detach().numpy()/log_interval
            losses.append(avg_loss)
            print('\nIter {}: Average loss for (pretrained) reduced Hamiltonian network: {:.6f}'.format(iter+1, avg_loss))
            total_loss = 0
            iter += 1

    return losses

def main(N_iter, device=device_default, lr=1e-3, log_interval=5):
    # Architecture setup and training.
    actor_net = Mlp(input_dim=1, output_dim=5, layer_dims=[16]).to(device)
    t = [0, 1, 2, 4, 8] #[0, 1, 2, 3, 4]
    print('Start training...\n')
    losses = train(actor_net, t, N_iter=N_iter, lr=lr, log_interval=log_interval)
    plt.yscale("log")
    plt.plot(np.arange(len(losses)), np.array(losses), 'bo-', linewidth=2)
    plt.title('Training losses versus iterations')
    plt.show()
    plt.clf()

    # Testing
    for _ in range(2):
        x0 = torch.ones(1) * np.random.randint(low=1, high=5)
        x0 = x0.to(device).reshape((1, x0.shape[0]))
        u = actor_net(x0)
        x = dynamic_blackbox(u, t, x0)
        x_val = []; t_val = []
        for e in x:
            x_val.append(e.cpu().detach().numpy()[0][0])
        for e in t:
             t_val.append(e.cpu().detach().numpy())
        x_val = np.array(x_val, dtype=float); t_val = np.array(t_val, dtype=float)
        plt.plot(t_val, x_val, 'ro-', linewidth=2)
        plt.title('Position over time of car agent with PMP trained policy')
        plt.show()
        plt.clf()

main(50000, log_interval=500)
print('\nRunning time is {:.2f} seconds'.format(time.time() - start_time))