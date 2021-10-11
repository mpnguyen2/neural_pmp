import numpy as np
import cv2

import torch
from torchdiffeq import odeint_adjoint as odeint

from common.common_nets import Mlp
from common.utils import generate_coords, spline_interp, isoperi_reward_from_img

from model_nets import HDNet

def test(h_layer_dims, model_path='models/hd.pth', out_file='videos/test8.wmv', 
         num_step=10, log_interval=1000):
    
    
    
    
    
'''
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