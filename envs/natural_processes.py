import numpy as np
from envs.base_env import ContinuousEnv

class ProdAndConsume(ContinuousEnv):
    def __init__(self, q_dim=1, u_dim=1, k=0.2, x0=20):
        super().__init__(q_dim, u_dim)
        self.k = k
        self.screen_width = 500
        self.screen_height = 500
        self.x0 = x0
    
    # (q0, q1) = (position, velocity)
    def f(self, q, u):
        u = np.clip(u, 0, 1)
        return self.k*q*u
        
    def f_u(self, q):
        val = self.k*q
        return val.reshape(-1, 1, 1)
    
    def L(self, q, u):
        return -((1-u)*q).reshape(-1)
    
    def g(self, q):
        return np.zeros(q.shape[0]) #-q.reshape(-1)
    
    def eval(self, q):
        return -q.reshape(-1)
    
    def eval_all(self, q_all):
        u = (q_all[:, 1:] - q_all[:, :-1])/(q_all[:, :-1]*self.k)
        return np.sum((1-u)*q_all[:, :-1])
    
    def sample_q(self, num_examples, mode='train'):
        return self.x0*(1 + np.random.uniform(high=1, low=-1, size=(num_examples, 1)))
    
class BeeColony(ContinuousEnv):
    def __init__(self, q_dim=2, u_dim=1,
                 d1=0.2, d2=0.3, b1=2.0, b2=1.0,
                 w0=10, qu0=0.5):
        super().__init__(q_dim, u_dim)
        self.screen_width = 500
        self.screen_height = 500
        self.d1, self.d2 = d1, d2
        self.b1, self.b2 = b1, b2
        self.w0, self.qu0= w0, qu0
    
    # (q0, q1) = (position, velocity)
    def f(self, q, u):
        w = q[:, 0]
        qu = q[:, 1]
        w_dot = -self.d1*w + self.b1*u[:,0]*w
        qu_dot = -self.d2*qu + self.b2*(1-u[:,0])*w
        return np.array([w_dot, qu_dot]).swapaxes(0, 1)
        
    def f_u(self, q):
        w = q[:, 0]
        c1 = self.b1*w
        c2 = -self.b2*w
        val = np.array([c1, c2]).swapaxes(0, 1)
        return val.reshape(-1, 2, 1)
    
    def L(self, q, u):
        return np.zeros(q.shape[0])  + 1000 * ((u > 1).reshape(-1).astype(float) + (u < 0).reshape(-1).astype(float))
    
    def g(self, q):
        return -q[:, 1]
    
    def eval(self, q):
        return -q[:, 1]
    
    def sample_q(self, num_examples, mode='train'):
        if mode == 'train':
            w = self.w0*(1 + np.random.uniform(high=1, low=-1, size=num_examples))
            qu = self.qu0*(1 + np.random.uniform(high=1, low=-1, size=num_examples))
        else:
            w = self.w0*(1 + np.random.uniform(high=1, low=-1, size=num_examples))
            qu = self.qu0*(1 + np.random.uniform(high=1, low=-1, size=num_examples))
        return np.array([w, qu]).swapaxes(0, 1)

def growth_fct(s, K_s, mu_max):
    return mu_max * s/(K_s + s)

class BioReactor(ContinuousEnv):
    # Bio-reactor from micro-algae to methane
    def __init__(self, q_dim=3, u_dim=1,
                 c_light=1.0, r=0.3,
                 beta=1.0, gamma=1.0,
                 mu_max=1.0, Ks=0.1,
                 y0=4):
        super().__init__(q_dim, u_dim)
        self.screen_width = 500
        self.c_light = c_light
        self.r = r
        self.beta = beta
        self.gamma = gamma
        self.mu_max = mu_max
        self.Ks = Ks
        self.y0 = y0
    
    # (q0, q1) = (position, velocity)
    def f(self, q, u):
        u = np.clip(u, 0, 1)
        u = u.reshape(-1)
        y = q[:, 0] # Concentration of biomass (micro-algae)
        s = q[:, 1] # Concentration of substrate
        x = q[:, 2] # Concentration of biomass (methane)
        y_dot = self.c_light * y/(1.0+y) - self.r*y - u*y
        s_dot = -growth_fct(s, self.Ks, self.mu_max) + u*self.beta * (self.gamma*y-s)
        x_dot = (growth_fct(s, self.Ks, self.mu_max) - u*self.beta) * x

        return np.array([y_dot, s_dot, x_dot]).swapaxes(0, 1)
        
    def f_u(self, q):
        y = q[:, 0] # Concentration of biomass (micro-algae)
        s = q[:, 1] # Concentration of substrate
        x = q[:, 2] # Concentration of biomass (methane)
        c1 = -y
        c2 = self.beta * (self.gamma*y-s)
        c3 = -self.beta * x
        val = np.array([c1, c2, c3]).swapaxes(0, 1)
        return val.reshape(-1, 3, 1)
    
    def L(self, q, u):
        s = q[:, 1]
        x = q[:, 2]
        return -growth_fct(s, self.Ks, self.mu_max)*x
    
    def g(self, q):
        s = q[:, 1]
        x = q[:, 2]
        return -growth_fct(s, self.Ks, self.mu_max)*x
    
    def eval(self, q):
        s = q[:, 1]
        x = q[:, 2]
        return -growth_fct(s, self.Ks, self.mu_max)*x
    
    def sample_q(self, num_examples, mode='train'):
        y = self.y0*(1 + np.random.uniform(high=1, low=-1, size=num_examples))
        s = np.zeros(num_examples)
        x = np.zeros(num_examples)
        return np.array([y, s, x]).swapaxes(0, 1)