import numpy as np
import cv2
import torch

from classical_controls import ContinuousEnv
from common.utils import spline_interp

def generate_coords(dim=16, num_samples=1024, xk=None, yk=None, xg=None, yg=None, total_random=True):
    if xk is None:
        xk, yk = np.mgrid[-1:1:4j, -1:1:4j]
    if xg is None:
        xg, yg = np.mgrid[-1:1:50j, -1:1:50j]
    if total_random:
        # Totally random
        q = np.random.rand(num_samples, dim) 
    else:
        # Random with zero padding
        width = int(np.sqrt(dim))
        q = np.zeros((num_samples, width, width)) 
        q[:, 1:(width-1), :(width-1)] = np.random.rand(num_samples, width-2, width-1)
        q = q.reshape(num_samples, -1)
    q -= .5
    
    return torch.tensor(q, dtype=torch.float)

def isoperi_reward_from_img(img):
    # Extract contours and calculate perimeter/area   
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0; peri = 0
    for cnt in contours:
        area -= cv2.contourArea(cnt, oriented=True)
        peri += cv2.arcLength(cnt, closed=True)
    if peri == 0:
        return 0
    return np.sqrt(abs(area))/abs(peri)

def isoperi_reward(xk, yk, z, xg, yg):
    img = spline_interp(xk, yk, z, xg, yg)
    return isoperi_reward_from_img(img)

class DensityOpt(ContinuousEnv):
    def __init__(self, q_dim=16, u_dim=16):
        super().__init__(q_dim, u_dim)
        self.xk, self.yk = np.mgrid[-1:1:4j, -1:1:4j]
        self.xg, self.yg = np.mgrid[-1:1:50j, -1:1:50j]
        
    # qdot = u. Shape is controlled solely by its adjustable velocity u
    def f(self, q, u):
        return u
    
    def f_u(self, q):
        return np.array([np.eye(q.shape[1])]*q.shape[0])
    
    # Lagrangian or running cost L
    def L(self, q, u):
        return 0.5*np.sum(u**2, axis=1)
    
    # Terminal cost g
    def g(self, q):
        return isoperi_reward(self.xk, self.yk, q, self.xg, self.yg)
        
            
'''
def test(h_layer_dims, model_path='models/hd.pth', out_file='videos/test8.wmv', 
         num_step=10, log_interval=1000):
    
# Setup fixed knots and grids
    xk, yk = np.mgrid[-1:1:4j, -1:1:4j]
    xg, yg = np.mgrid[-1:1:50j, -1:1:50j]
    # Setup video writer
    
    
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