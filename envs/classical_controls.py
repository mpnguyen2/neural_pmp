### Continuous version for Hamiltonian dynamics training of environments in OpenAI Gym classical controls

import numpy as np
import math
from gym.utils import seeding

# Generic continous environment for reduced Hamiltonian dynamics training
class ContinuousEnv():
    def __init__(self, q_dim=1, u_dim=1):
        self.q_dim = q_dim
        self.u_dim = u_dim
        self.eps = 1e-8
        self.id = np.eye(q_dim)
    
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
    def sample_q(self, num_examples):
        return np.zeros((num_examples, self.q_dim))
        
    def render():
        return

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

class Pendulum(ContinuousEnv):
    def __init__(self, q_dim=2, u_dim=1, gravity=9.8):
        super().__init__(q_dim, u_dim)
        self.max_speed = 8
        self.max_torque = 2.0

        self.gravity = gravity
        self.m = 1.0
        self.l = 1.0
    
    # (q0, q1) = (position, velocity)
    def f(self, q, u):
        u = np.clip(u, -self.max_torque, self.max_torque)
        gravity, m, l = self.gravity, self.m, self.l
        return np.concatenate((q[:, 1:], 3*gravity/(2*l)*np.sin(q[:, 0:1]) + 3/(m*(l**2))*u[:, 0:]), axis=1)
        
    def f_u(self, q):
        m, l = self.m, self.l
        N = q.shape[0]
        return np.concatenate((np.zeros((N, 1, 1)), 
                               3/(m*(l**2))*np.ones((N, 1, 1))), axis=1)
    
    def L(self, q, u):
        u = np.clip(u, -self.max_torque, self.max_torque)
        return angle_normalize(q[:, 0])**2 + 0.1*q[:, 1]**2 + 0.001 * (u[:, 0]**2)
    
    def g(self, q):
        return angle_normalize(q[:, 0])**2 + 0.1*q[:, 1]**2
    
    def sample_q(self, num_examples):
        return np.concatenate(
            (np.random.uniform(high=np.pi, low=-np.pi, size=(num_examples, 1)),
            np.random.uniform(high=1, low=-1, size=(num_examples, 1))),
            axis=1)
    
class MountainCar(ContinuousEnv):
    def __init__(self, q_dim=2, u_dim=1, goal_velocity=0):
        super().__init__(q_dim, u_dim)
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = (
            0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        )
        self.goal_velocity = goal_velocity
        self.power = 0.0015
    
    # (q0, q1) = (position, velocity)
    def f(self, q, u):
        force = np.clip(u, self.min_action, self.max_action)
        return np.concatenate((q[:, 1:], 
                force[:, 0:]*self.power - 0.0025 * np.cos(3 * q[:, 0:1])), axis=1)
    
    def f_u(self, q):
        N = q.shape[0]
        return np.concatenate((np.zeros((N, 1, 1)), np.ones((N, 1, 1))*self.power), axis=1)
    
    def L(self, q, u):
        return u[:, 0]**2

    def g(self, q):
        return (self.goal_position-q[:, 0])**2 + (self.goal_velocity-q[:, 1])**2
    
    def sample_q(self, num_examples):
        return np.concatenate(
            (np.random.uniform(high=self.max_position/2, low=self.min_position/2, size=(num_examples, 1)),
            np.random.uniform(high=self.max_speed/2, low=-self.max_speed/2, size=(num_examples, 1))),
            axis=1)
    
class CartPole(ContinuousEnv):
    def __init__(self, q_dim=4, u_dim=1):
        super().__init__(q_dim, u_dim)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
    
        # For continous version
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4   
        self.seed()

        
    def f(self, q, u):
        _, x_dot, theta, theta_dot = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        force = self.force_mag * np.clip(u, -1, 1).reshape(-1)
        costheta, sintheta = np.cos(theta), np.sin(theta)
        temp = (
            force + self.polemass_length * (theta_dot ** 2) * sintheta
        ) / self.total_mass
        
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        thetaacc, xacc = thetaacc.reshape(-1, 1), xacc.reshape(-1, 1), 
    
        return np.concatenate((x_dot.reshape(-1, 1), xacc, theta_dot.reshape(-1, 1), thetaacc), axis=1)
    
    def f_u(self, q):
        theta = q[:, 2]
        N = q.shape[0]
        costheta = np.cos(theta)
        tmp_u = 1/self.total_mass * self.force_mag
        xacc_u = tmp_u * np.ones((N, 1))
        thetaacc_u = -costheta*tmp_u/(
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        thetaacc_u = thetaacc_u.reshape(-1, 1)
        return np.concatenate((np.zeros((N, 1)), xacc_u, np.zeros((N, 1)), thetaacc_u), axis=1)\
            .reshape(-1, self.q_dim, self.u_dim)
        
        
    def L(self, q, u):
        return (q[:, 0] - self.x_threshold/2)**2 + (q[:, 2] - self.theta_threshold_radians/2)**2 + 0.5*u**2
    
    def g(self, q):
        return (q[:, 0] - self.x_threshold/2)**2 + (q[:, 2] - self.theta_threshold_radians/2)**2
    
    def sample_q(self, num_examples):
        return np.random.uniform(low=-0.05, high=0.05, size=(num_examples, 4))