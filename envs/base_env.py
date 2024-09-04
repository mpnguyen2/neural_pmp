import numpy as np
from gym.utils import seeding

### Generic continuous environment for reduced Hamiltonian dynamics framework
class ContinuousEnv():
    def __init__(self, q_dim=1, u_dim=1):
        self.q_dim = q_dim
        self.u_dim = u_dim
        self.eps = 1e-8
        self.id = np.eye(q_dim)
        self.seed()
        
        # Viewer for rendering image
        self.viewer = None
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # Dynamics f
    def f(self, q, u):
        return np.zeros((q.shape[0], self.q_dim))
    
    # Partial derivative of dynamics f wrt control u. Assuming linear control
    def f_u(self, q):
        return np.zeros((q.shape[0], self.q_dim, self.u_dim))
    
    # Lagrangian or running cost L
    def L(self, q, u):
        return np.zeros(q.shape[0])
    
    # Terminal cost g
    def g(self, q):
        return np.zeros(q.shape[0])
    
    # Nabla of g
    def nabla_g(self, q):
        ret = np.zeros((q.shape[0], self.q_dim))
        for i in range(self.q_dim):
            ret[:, i] = (self.g(q+self.eps*self.id[i])-self.g(q-self.eps*self.id[i]))/(2*self.eps)
        return ret
    
    # Sampling state q
    def sample_q(self, num_examples, mode='train'):
        return np.zeros((num_examples, self.q_dim))
    
    # Image rendering
    def render(self, q, mode="rgb_array"):
        return
    
    # Close rendering
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None