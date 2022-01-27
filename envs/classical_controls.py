### Continuous version for Hamiltonian dynamics training of environments in OpenAI Gym classical controls
### The rendering part is the same as OpenAI Gym

import numpy as np
import math
from gym.utils import seeding
from os import path

### Generic continuous environment for reduced Hamiltonian dynamics framework
class ContinuousEnv():
    def __init__(self, q_dim=1, u_dim=1, control_coef=0.5):
        self.q_dim = q_dim
        self.u_dim = u_dim
        self.control_coef = control_coef
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
        return self.control_coef*np.sum(u**2, axis=1) + self.g(q)
    
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
   
#### Mountain car for PMP ####
class MountainCar(ContinuousEnv):
    def __init__(self, q_dim=2, u_dim=1, control_coef=0.5, goal_velocity=0):
        super().__init__(q_dim, u_dim, control_coef)
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
        return self.control_coef*(u[:, 0]**2) + self.g(q)
    
    def g(self, q):
        return np.maximum(0, self.goal_velocity-q[:, 1]) + np.maximum(0, self.goal_position-q[:, 0])
    
    def sample_q(self, num_examples, mode='train'):
        if mode == 'train': 
            a = 0.5
        else:
            a = 1
        return np.concatenate(
            (a*np.random.uniform(high=self.max_position, low=self.min_position, size=(num_examples, 1)),
            np.zeros((num_examples, 1))),
            axis=1)
    
    def criteria_q(self, q):
        return (self.goal_position-q[0])**2
    
    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55
    
    def render(self, q, mode="rgb_array"):
        # Set position and velocity boundary
        if q[0] >= self.goal_position:
            q[0] = self.goal_position
        if q[1] >= self.goal_velocity:
            q[1] = self.goal_velocity
            
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos = q[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

#### CartPole for PMP ####
class CartPole(ContinuousEnv):
    def __init__(self, q_dim=4, u_dim=1, control_coef=0.5):
        super().__init__(q_dim, u_dim, control_coef)
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
        
    def f(self, q, u):
        _, x_dot, theta, theta_dot = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        force = self.force_mag * u.reshape(-1)
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
        tmp_u = self.force_mag /self.total_mass
        xacc_u = tmp_u * np.ones((N, 1))
        thetaacc_u = -costheta*tmp_u/(
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        thetaacc_u = thetaacc_u.reshape(-1, 1)
        return np.concatenate((np.zeros((N, 1)), xacc_u, np.zeros((N, 1)), thetaacc_u), axis=1)\
            .reshape(-1, self.q_dim, self.u_dim)
        
        
    def L(self, q, u):
        return self.control_coef*(np.sum(u**2, axis=1)) - q[:, 1]**2
    
    def g(self, q):
        #noise = np.random.normal(scale=0.001, size=(q.shape[0]))
        #t = [self.x_threshold/2, self.theta_threshold_radians/2]
        #a = 0.005
        return (q[:, 2]/self.theta_threshold_radians)**2 #(a**2-q[:, 0]**2)
    
    def sample_q(self, num_examples, mode='train'):
        if mode == 'train':
            a = 0.01
        else:
            a = 0.05
        return np.random.uniform(low=-a, high=a, size=(num_examples, 4))
    
    def render(self, q, mode="rgb_array"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        cartx = q[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-q[2])

        return self.viewer.render(return_rgb_array=mode=="rgb_array")
    
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

#### Pendulum for PMP ####
class Pendulum(ContinuousEnv):
    def __init__(self, q_dim=2, u_dim=1, control_coef=0.5, gravity=9.8):
        super().__init__(q_dim, u_dim, control_coef)
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
        return self.control_coef*u[:, 0]**2 + self.g(q)
    
    def g(self, q):
        return (angle_normalize(q[:, 0])+np.pi/2)**2
    
    def sample_q(self, num_examples, mode='train'):
        if mode=='train':
            a = 0.1
        else:
            a = 0.01
        return a*np.concatenate(
                (np.random.uniform(high=np.pi, low=-np.pi, size=(num_examples, 1)),
                np.random.uniform(high=1, low=-1, size=(num_examples, 1))),
                axis=1)
        
    def render(self, q, mode="rgb_array"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            #self.imgtrans = rendering.Transform()
            #self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(q[0]+np.pi/2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")


class TestEnv(ContinuousEnv):
    def __init__(self, q_dim=1, u_dim=1, control_coef=0.5):
        super().__init__(q_dim, u_dim, control_coef)
        # Viewer for rendering image
        #self.viewer = None
    
    # Dynamics f
    def f(self, q, u):
        return 2*(1-u)
    
    # Partial derivative of dynamics f wrt control u. Assuming linear control
    def f_u(self, q):
        return -2*np.ones((q.shape[0], self.q_dim, self.u_dim))
    
    # Terminal cost g
    def g(self, q):
        return -q.reshape(q.shape[0])
    
    # Sampling state q
    def sample_q(self, num_examples, mode='train'):
        if mode == 'train':
            scaling = 1
            return scaling*(np.random.rand(num_examples, self.q_dim)-0.5)
        return np.ones((num_examples, self.q_dim))