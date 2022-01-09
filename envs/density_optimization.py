import numpy as np
import cv2
import torch
import gym

from shapely.geometry import Polygon
from scipy.interpolate import CubicSpline

from envs.classical_controls import ContinuousEnv   
from common.utils import spline_interp

from collections import namedtuple
ImgDim = namedtuple('ImgDim', 'width height')

# Shape optimization for boundary parametrization method
class DensityOptBoundary(ContinuousEnv):
    def __init__(self, q_dim=16, u_dim=16, control_coef=0.5):
        super().__init__(q_dim, u_dim, control_coef)
        self.num_coef = q_dim//2
        self.ts = np.linspace(0, 1, 80)
        
    # qdot = u. Shape is controlled solely by its adjustable velocity u
    def f(self, q, u):
        return u
    
    def f_u(self, q):
        return np.array([np.eye(q.shape[1])]*q.shape[0])
    
    # Terminal cost g
    def g(self, q):
        ret = np.zeros(q.shape[0])
        for i in range(q.shape[0]):
            cs = CubicSpline(np.linspace(0,1,self.num_coef), q[i].reshape(2, self.num_coef).T)
            coords = cs(self.ts)
            polygon = Polygon(zip(coords[:,0], coords[:,1]))
            ret[i] = polygon.length/np.sqrt(polygon.area)
        
        return ret
          
    def sample_q(self, num_examples, mode='train', shape='random'):
        #if mode == 'train':
        #    return 0.4*(np.random.rand(num_examples, self.q_dim) - .5)

        # For test, either sample from a random shape with above/below parts or ellipse or square
        return self.sample_distinct_q(num_examples, shape=shape)
    
    def sample_distinct_q(self, num_examples, shape='random'):
        qs = np.zeros((num_examples, self.q_dim))
        q = np.zeros(self.q_dim)
        for i in range(num_examples):
            if shape == 'ellipse':
                    t = np.arange(self.num_coef)/self.num_coef
                    q[:self.num_coef] = 0.1*np.sin(2*np.pi*t)
                    q[self.num_coef:] = np.cos(2*np.pi*t)
            elif shape == 'square':
                # Assume n%4 == 0
                n = self.num_coef//4
                # x-coord
                q[:n] = np.arange(n)/n
                q[n:2*n] = 1
                q[2*n:3*n] = 1 - (np.arange(n)/n)
                q[3*n:4*n] = 0
                # y-coord
                q[4*n:5*n] = 0
                q[5*n:6*n] = np.arange(n)/n
                q[6*n:7*n] = 1
                q[7*n:8*n] = 1 - (np.arange(n)/n)
            elif shape == 'random':
                # Assume n%2 == 0
                n = self.num_coef//2
                # x-coord
                q[0:n] = 0.8*self.np_random.rand(n) + 0.2
                q[n:2*n] = -0.8*self.np_random.rand(n) - 0.2
                # y-coord
                q[2*n:3*n] = np.arange(n)/n
                q[3*n:4*n] = np.arange(n)/n
            
            qs[i, :] = q
        
        return qs
        
    def render(self, q, mode="rgb_array"):
        screen_width = 600
        screen_height = 600
        eps = 1e-5
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
        
        # Use cubic spline to smooth out the new state parametric curve
        cs = CubicSpline(np.linspace(0,1,self.num_coef), q.reshape(2, self.num_coef).T)
        coords = cs(self.ts)
        coords = coords/(np.max(np.abs(coords))+eps)*100 + 300
        verts = zip(coords[:,0], coords[:,1])
        
        self.viewer.draw_polygon(list(verts))

        if q is None:
            return None

        return self.viewer.render(return_rgb_array=mode=='rgb_array')
     
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# Shape optimization for level-set method
class DensityOpt(ContinuousEnv):
    def __init__(self, q_dim=16, u_dim=16, control_coef=0.5):
        super().__init__(q_dim, u_dim, control_coef)
        self.xk, self.yk = np.mgrid[-1:1:4j, -1:1:4j]
        self.xg, self.yg = np.mgrid[-1:1:50j, -1:1:50j]
        self.viewer = ImgDim(width=self.xg.shape[0], height=self.yg.shape[1])

    # qdot = u. Shape is controlled solely by its adjustable velocity u
    def f(self, q, u):
        return u
    
    def f_u(self, q):
        return np.array([np.eye(q.shape[1])]*q.shape[0])
    
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
            return generate_coords(num_samples=num_examples, total_random=False, r1=6, r2=9)
        
    def render(self, q, mode="rgb_array"):
        xk, yk, xg, yg = self.xk, self.yk, self.xg, self.yg
        return spline_interp(q.reshape(xk.shape[0], yk.shape[0]), xk, yk, xg, yg)

# Helper functions
def generate_coords(dim=16, num_samples=1024, xk=None, yk=None, xg=None, yg=None, 
                    r1=-100, r2=100, total_random=False, fixed=True):
    if xk is None:
        xk, yk = np.mgrid[-1:1:4j, -1:1:4j]
    if xg is None:
        xg, yg = np.mgrid[-1:1:50j, -1:1:50j]
    width = int(np.sqrt(dim))
    qs = np.zeros((num_samples, dim))
    cnt = 0
    '''
    if fixed:
        q = np.zeros((width, width))
        q[1:2, 1:width-1] = np.ones((1, width-2))
        q[2:3, width//2] = 1;
        #q[width//2, width//2] = 0
        q = q.reshape(-1)
        for i in range(num_samples):
            qs[i] = q
        cnt = num_samples
    '''  
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