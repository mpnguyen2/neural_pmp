import numpy as np
import math
from os import path
from envs.classical_controls import ContinuousEnv
from gym import utils
from gym.utils import seeding
from gym.envs.mujoco.mujoco_env import MuJocoPyEnv
from gym.spaces import Box

#### Ant Environment from Mujoco Gym ####
# To install Mujoco, go to https://github.com/openai/mujoco-py#install-mujoco
class Ant(MuJocoPyEnv, utils.EzPickle, ContinuousEnv):
    # Metadata is directly from Mujoco library
    metadata ={
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 20,
    }
    def __init__(self, q_dim=111, u_dim=1, control_coef=0.5):
        self.q_dim = 111
        self.u_dim = 1
        observation_space = Box(low=-np.inf, high=np.inf, shape=(111,), dtype=np.float64)
        MuJocoPyEnv.__init__(self, "ant.xml", 5, observation_space=observation_space)
        utils.EzPickle.__init__(self)
        super().__init__("ant.xml", 5, observation_space, q_dim, u_dim, control_coef)
        self.goal_position = np.array([25, 25]) # XY of goal, no Z

    # Dynamics f    
    def f(self, q, u):
        # qs
        self.xposbefore = self.get_body_com("torso")[0]
        self.vbefore = self.sim.data.qvel.flat
        self.do_simulation(u, self.frame_skip) # Does simulation
        self.xposafter = self.get_body_com("torso")[0]
        self.vafter = self.sim.data.qvel.flat
        # obs: 
        #   * qpos (12): XYZ and orientation (XYZW) of torso (4) and joint angles (8)
        #   * qvel (14): torso velocities (3:xyz), angular velocitiries (3:xyz) joint velocities (8)
        #   * cfrc_ext (14*6): external forces (force x,y,z and torque x,y,z) applied to each of the links at the center of mass.
        #       * the ground link, the torso link, and 12 links for all legs (3 links for each leg).
        obs = np.concatenate((self.sim.data.qpos.flat, self.sim.data.qvel.flat,), axis=1)
        return obs

    #  Partial derivative of dynamics f wrt control u. Assuming linear control
    def f_u(self, q):
        dq_u = self.xposafter - self.xposbefore # q
        dv_u = self.vbefore -  self.vafter
        return np.concatenate((dq_u, dv_u), axis=1)

    # Lagrangian or running cost L
    def L(self, q, u):
        self.ctrl_cost = 0.5 * np.square(u).sum() # Control cost
        self.forward_reward = (self.xposafter - self.xposbefore) / self.dt # Reward for velocity gained (dPosition)
        self.contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        # Distance from goal (goal-current XY) - original
        return -(self.forward_reward - self.ctrl_cost - self.contact_cost + 1)

    # Terminal cost
    def g(self, q):
        zposition = self.state_vector()[2]
        terminal = (not (0.2 <= zposition <= 1.0)) and np.isfinite(self.state_vector())
        return np.linalg.norm(self.sim.data.qpos.flat[:3]-self.goal_position) 

    def sample_q(self, num_examples, mode='train'):
        if mode == 'train': 
            a = 0.5
        else:
            a = 1
        return np.concatenate(
            (a*np.random.uniform(high=50, low=-50, size=(num_examples, 1)),
            np.zeros((num_examples, 1))),
            axis=1)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self.f()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5
