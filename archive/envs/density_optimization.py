import numpy as np
import cv2
import torch

from envs.classical_controls import ContinuousEnv
   
from common.utils import spline_interp

from collections import namedtuple
ImgDim = namedtuple('ImgDim', 'width height')

class DensityOpt(ContinuousEnv):
    def __init__(self, q_dim=16, u_dim=16):
        super().__init__(q_dim, u_dim)
        self.xk, self.yk = np.mgrid[-1:1:4j, -1:1:4j]
        self.xg, self.yg = np.mgrid[-1:1:50j, -1:1:50j]
        self.viewer = ImgDim(width=self.xg.shape[0], height=self.yg.shape[1])

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
        cost = np.zeros(q.shape[0])
        for i in range(q.shape[0]):
            cost[i] = isoperi_cost(q[i], self.xk, self.yk, self.xg, self.yg)
        return cost
        
    def sample_q(self, num_examples, mode='train'):
        if mode=='train':
            return generate_coords(num_samples=num_examples, total_random=False, r1=3, r2=7)
        else:
            return generate_coords(num_samples=num_examples, total_random=True, r1=6, r2=8)
        
    def render(self, q, mode="rgb_array"):
        xk, yk, xg, yg = self.xk, self.yk, self.xg, self.yg
        return spline_interp(q.reshape(xk.shape[0], yk.shape[0]), xk, yk, xg, yg)

# Helper functions
def generate_coords(dim=16, num_samples=1024, xk=None, yk=None, xg=None, yg=None, 
                    r1=-100, r2=100, total_random=False):
    if xk is None:
        xk, yk = np.mgrid[-1:1:4j, -1:1:4j]
    if xg is None:
        xg, yg = np.mgrid[-1:1:50j, -1:1:50j]
    width = int(np.sqrt(dim))
    qs = np.zeros((num_samples, dim))
    cnt = 0
    while cnt < num_samples:
        if total_random:
            q = np.random.rand(width, width)
        # Random with zero padding
        else:
            q = np.zeros((width, width)) 
            q[1:(width-1), :(width-1)] = np.random.rand(width-2, width-1)
        q = q.reshape(-1)
        c = isoperi_cost(q-.5, xk, yk, xg, yg) 
        if c > r1 and c < r2:
            qs[cnt, :] = q
            cnt += 1
                
    qs -= .5
    
    return qs

def isoperi_cost_from_img(img):
    # Extract contours and calculate perimeter/area   
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0; peri = 0
    for cnt in contours:
        area -= cv2.contourArea(cnt, oriented=True)
        peri += cv2.arcLength(cnt, closed=True)
    if area == 0:
        return 100
    return np.abs(peri)/np.sqrt(abs(area))

def isoperi_cost(z, xk, yk, xg, yg):
    img = spline_interp(z, xk, yk, xg, yg)
    return isoperi_cost_from_img(img)